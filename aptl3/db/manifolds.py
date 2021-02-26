import itertools
import json
import logging
import os
import signal
import sys
import threading
import traceback
from collections import Counter
from datetime import datetime
from tempfile import mkstemp
from time import perf_counter
from typing import Tuple, Optional, Iterable, Dict, Any

from annoy import AnnoyIndex

from .embeddings import EmbeddingsDatabase
from .locations import LocationsDatabase
from ..am import ActorSystem, Actor
from ..embedding.abc import RequestIgnored


class ManifoldsDatabase(LocationsDatabase, EmbeddingsDatabase):
    """
    Storage and ANN indexing of high-dimensional vector data ("manifold items")
    abstracting away the index implementation and the possible multitude of indexes for one given embedding model.
    """

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self.__logger = logging.getLogger(__name__)
        self.__manifolds_embedding: Dict[int, int] = dict()
        self.__manifolds_annoy_index: Dict[int, AnnoyIndex] = dict()
        self.__manifolds_background_build_threads: Dict[
            int, Tuple[threading.Thread, threading.Event, threading.Event]] = dict()
        self.__manifolds_bg_monitor_thread: Optional[threading.Thread] = None

    def _get_manifold_embedding_id(self, manifold_id: int) -> int:
        try:
            return self.__manifolds_embedding[manifold_id]
        except KeyError:
            c = self._db.execute('SELECT embedding_id FROM Manifolds WHERE manifold_id = ?', (manifold_id,))
            row = c.fetchone()
            if row is None:
                raise ValueError(manifold_id)  # it does not exist.
            embedding_id: int = row[0]
            self.__manifolds_embedding[manifold_id] = embedding_id
            return embedding_id

    def _get_manifold_metadata(self, manifold_id: int) -> dict:
        c = self._db.execute('SELECT metadata FROM Manifolds WHERE manifold_id = ?', (manifold_id,))
        row = c.fetchone()
        if row is None:
            raise ValueError(manifold_id)  # does not exist.
        metadata: dict = json.loads(row[0])
        return metadata

    def _get_annoy_index(self, manifold_id: int) -> AnnoyIndex:
        try:
            return self.__manifolds_annoy_index[manifold_id]
        except KeyError:
            c = self._db.execute(
                'SELECT metadata, dim '
                'FROM Manifolds JOIN Embeddings ON Manifolds.embedding_id = Embeddings.embedding_id '
                'WHERE manifold_id = ?', (manifold_id,))
            res = c.fetchone()
            if res is None:
                raise ValueError(manifold_id)  # it does not exist.
            metadata, dim = res

            metadata = json.loads(metadata)
            fn = metadata['fn']
            from annoy import AnnoyIndex
            index = AnnoyIndex(dim, metadata['metric'])
            index.load(os.path.join(self.get_data_dir(), fn))
            self.__manifolds_annoy_index[manifold_id] = index
            return index

    def get_item_vector(self, manifold_id: int, item_i: int):
        index = self._get_annoy_index(manifold_id)
        return index.get_item_vector(item_i)

    def get_media_vectors(self, media_id: bytes) -> Iterable[Tuple[int, Any]]:
        """Yields embedding vector data for the given media, stored in any ready manifold."""
        c = self._db.execute(
            'SELECT M.embedding_id, MI.manifold_id, MI.item_i '
            'FROM ManifoldItems MI NATURAL JOIN Manifolds M '
            'WHERE M.ready AND MI.media_id = ? AND MI.item_i IS NOT NULL', (media_id,))
        for embedding_id, manifold_id, item_i in c:
            yield embedding_id, self.get_item_vector(manifold_id, item_i)

    def find_vector_neighbours(
            self, embedding_id: int, vector, *,
            n: int, search_k: int = -1,
    ) -> Iterable[Tuple[int, int, float]]:
        c = self._db.execute(
            'SELECT M.manifold_id '
            'FROM Manifolds M JOIN Embeddings E ON M.embedding_id = E.embedding_id '
            'WHERE M.ready AND E.ready AND E.embedding_id = ? '
            'ORDER BY M.manifold_id DESC ', (embedding_id,))
        for manifold_id, in c:
            try:
                index = self._get_annoy_index(manifold_id)
                out = index.get_nns_by_vector(vector, n=n, search_k=search_k, include_distances=True)
                if out is None:
                    raise RuntimeError
                items, dis = out
            except Exception:
                self.__logger.debug(f"Could not use manifold #{manifold_id}", exc_info=True)
                continue
            else:
                yield from zip(itertools.repeat(manifold_id), items, dis)

    def _make_new_manifold(self, *, embedding_id: int, metric: str = 'euclidean', dim: Optional[int] = None) -> int:
        """Will commit multiple write transactions"""
        if dim is None:
            embedding = self.get_embedding(embedding_id)
            dim = embedding.get_dim()

        self.begin_exclusive_transaction()
        c = self._db.execute('INSERT INTO Manifolds (embedding_id, building, metadata) VALUES (?, ?, ?);',
                             (embedding_id, False, '{}'))
        manifold_id = c.lastrowid

        self.__manifolds_embedding[manifold_id] = embedding_id

        full_fn = mkstemp(dir=self.get_data_dir(),
                          prefix=f'{manifold_id:06d}.',
                          suffix='.annoy', text=False)[1]

        index = AnnoyIndex(dim, metric)
        index.on_disk_build(full_fn)
        self.__manifolds_annoy_index[manifold_id] = index  # must put in cache, otherwise it will try to index.load(fn)

        fn = os.path.relpath(full_fn, self.get_data_dir())
        metadata = dict(
            fn=fn,
            metric=metric,
            utc=str(datetime.utcnow())
        )

        self._db.execute('UPDATE Manifolds SET building = ?, metadata = ? WHERE manifold_id = ?;',
                         (True, json.dumps(metadata), manifold_id))
        self.commit()
        return manifold_id

    def build_new_manifold(self, *, embedding_id: int, metric: str = 'euclidean', n_trees: int = 10,
                           notice: Optional[threading.Event] = None,
                           stop: Optional[threading.Event] = None,
                           ) -> Tuple[int, int]:
        """Will commit multiple write transactions"""
        with self._db:
            manifold_id = self._make_new_manifold(embedding_id=embedding_id, metric=metric)
            inserted = self._populate_manifold(manifold_id=manifold_id, notice=notice, stop=stop)
            if inserted == 0:
                self.begin_exclusive_transaction()
                self.execute('DELETE FROM Manifolds WHERE manifold_id = ?', (manifold_id,))
                self.commit()
                # NOTE: the index file is left as is and is not deleted.
                self.__logger.info(f'No items or holes inserted; Manifold {manifold_id:d} deleted.')
            else:
                self._build_manifold_index(manifold_id, n_trees)
            return manifold_id, inserted

    @staticmethod
    def _detached_build_new_manifold(data_dir: str, **kwargs) -> None:
        try:
            ManifoldsDatabase(data_dir).build_new_manifold(**kwargs)
        except Exception as ex:
            print(f'Error in detached manifold build: {type(ex).__name__}: {str(ex)}', file=sys.stderr)
            traceback.print_exc()

    def __manifolds_bg_monitor(self, parent_thread: threading.Thread):
        assert parent_thread is not threading.current_thread()
        parent_thread.join()
        # NOTE: there should be no other reference to `self`, so this should be safe even from a different thread.
        self.close_bg_manifold_builds()

    def notify_bg_manifold_build(self, *, embedding_id: Optional[int] = None):
        """Wakeup or start background manifold build for the given embedding, or all embeddings if id not given."""
        if embedding_id is None:
            for embedding_id in self.list_ready_embedding_ids():
                self.notify_bg_manifold_build(embedding_id=embedding_id)
            return
        try:
            th, notice, stop = self.__manifolds_background_build_threads[embedding_id]
        except KeyError:
            if self.__manifolds_bg_monitor_thread is None:
                self.__manifolds_bg_monitor_thread = threading.Thread(
                    target=self.__manifolds_bg_monitor,
                    args=(threading.current_thread(),),
                    daemon=True)
                self.__manifolds_bg_monitor_thread.start()
            notice, stop = threading.Event(), threading.Event()
            th = threading.Thread(
                target=self._detached_build_new_manifold,
                args=(self.get_data_dir(),),
                kwargs=dict(embedding_id=embedding_id, notice=notice, stop=stop),
                daemon=False)
            th.start()
            self.__manifolds_background_build_threads[embedding_id] = th, notice, stop
        else:
            notice.set()

    def close_bg_manifold_builds(self):
        for th, notice, stop in self.__manifolds_background_build_threads.values():
            stop.set()
            notice.set()
        for embedding_id, (th, notice, stop) in list(self.__manifolds_background_build_threads.items()):
            self.__logger.debug(f'Waiting for background manifold build with embedding {embedding_id}...')
            th.join()
            del self.__manifolds_background_build_threads[embedding_id]
        self.__logger.debug(f'All threads joined.')

    def _populate_manifold(self, *, manifold_id: int,
                           notice: Optional[threading.Event] = None,
                           stop: Optional[threading.Event] = None,
                           ) -> int:
        """Will commit multiple write transactions"""
        db = self
        embedding_id = self._get_manifold_embedding_id(manifold_id)
        embedding = self.get_embedding(embedding_id)
        index = self._get_annoy_index(manifold_id)

        stop = threading.Event() if stop is None else stop
        booked = 0
        enqueued = 0
        collected = 0
        written = 0
        computed = 0
        failures = 0
        ignored = 0
        pending = 0
        start_time = perf_counter()
        qsize = 256  # This has to be tuned depending on the embedding model performance
        assert qsize >= 8
        book_minimum = qsize // 4
        pending_limit = qsize
        write_minimum = qsize // 8

        # NOTE: You may notice the system not enqueuing enough jobs to exploit the maximum concurrency.
        #       This can happen after the initial wave of jobs go through the fastest station.
        #       This queue system is designed to be steady and avoids accumulating jobs at the slowest station.

        def _prefetch_jobs():
            return self.execute(
                'SELECT media_id, url '
                'FROM MediaLocations '
                'WHERE media_id IS NOT NULL '
                'AND media_id not in ('
                '    SELECT MI.media_id FROM ManifoldItems as MI '
                '    NATURAL JOIN Manifolds M WHERE M.embedding_id = :embedding_id) '
                'AND media_id not in ('
                '    SELECT MH.media_id FROM ManifoldHoles as MH '
                '    NATURAL JOIN Manifolds M WHERE M.embedding_id = :embedding_id) '
                'GROUP BY media_id '
                'ORDER BY MIN(ts_first_access);',
                dict(embedding_id=embedding_id))
            # Ordering should not be important, but will affect the insertion order of items
            # and thus it may affect the performance of some query on manifold items

        # noinspection PyUnusedLocal
        def _interrupt_handler(signum, frame):
            stop.set()
            if notice is not None:
                notice.set()
            print(f'\n  >>> SIGINT Stopping as soon as possible... ')

        self.__logger.info(f'Populating index for manifold {manifold_id}...')

        if threading.main_thread() == threading.current_thread():
            # Could only call signal in main thread
            _previous_interrupt_handler = signal.signal(signal.SIGINT, _interrupt_handler)

        with ActorSystem(maxsize=qsize*2) as actor_sys:

            (actor_sys
             .add_thread('fetch', _MaybeFetchActor(self.get_data_dir()))
             .add_thread('batch', _EmbeddingBatchingActor(embedding), pool=0)
             .add_thread('embed', _EmbeddingTransformActor(embedding))
             .add_thread('annoy', _AnnoyAddActor(index))
             .add_mailbox('main')
             .start())

            def _status(_s: str):
                throughput = computed / actor_sys.elapsed
                print(
                    f"\r{_s:20s} ({computed:d} {ignored:d} {failures:d}) {throughput=:5.2f}/s "
                    f"{actor_sys.status}    ",
                    end="\r", flush=True)

            _jobs_cursor = _prefetch_jobs()

            while True:
                items = []
                del_holes = []
                upd_holes = []

                def _some_pending_msgs():
                    n = 0
                    while pending - n > 0 and n < write_minimum:
                        yield actor_sys.ask('main')
                        n += 1
                    yield from actor_sys.ask_available('main')

                # Collect previous results
                _status('processing...')
                for media_id, ex, item_i in _some_pending_msgs():
                    collected += 1
                    if ex is None:
                        computed += 1
                        items += [(manifold_id, item_i, media_id)]
                        del_holes += [(manifold_id, media_id)]
                    else:
                        msg = f'{type(ex).__name__}: {str(ex)}'
                        if isinstance(ex, RequestIgnored):
                            ignored += 1
                        else:
                            failures += 1
                            tb = traceback.TracebackException.from_exception(ex)
                            print()
                            self.__logger.error(msg)
                            self.__logger.debug(''.join(tb.format()))
                        upd_holes += [(msg, manifold_id, media_id)]
                    _status('processing...')

                # Start db transaction
                _status('locking...')
                start_time += self.begin_exclusive_transaction()

                # Write previous results
                _status('writing...')
                db.executemany('INSERT INTO ManifoldItems (manifold_id, item_i, media_id) VALUES (?, ?, ?)', items)
                db.executemany('DELETE FROM ManifoldHoles WHERE manifold_id = ? AND media_id = ?', del_holes)
                db.executemany('UPDATE ManifoldHoles SET msg = ? WHERE manifold_id = ? AND media_id = ?', upd_holes)

                # Close cycle
                written += len(items) + len(upd_holes)
                pending -= len(items) + len(upd_holes)
                items.clear()
                del_holes.clear()
                upd_holes.clear()

                # Exclusively take a batch of media
                _status('reading...')
                if stop.is_set() or pending + book_minimum > pending_limit:
                    batch = []
                else:
                    batch = _jobs_cursor.fetchmany(pending_limit - pending)
                    for _i, (__media_id_, _) in reversed(list(enumerate(batch))):
                        row = self.execute(
                            'SELECT EXISTS('
                            ' SELECT media_id FROM ManifoldItems NATURAL JOIN Manifolds '
                            ' WHERE media_id = :media_id AND embedding_id = :embedding_id '
                            ' UNION '
                            ' SELECT media_id FROM ManifoldHoles NATURAL JOIN Manifolds '
                            ' WHERE media_id = :media_id AND embedding_id = :embedding_id '
                            ')', dict(media_id=__media_id_, embedding_id=embedding_id)).fetchone()
                        if row[0] == 1:
                            batch.pop(_i)
                            # NOTE: will not repopulate the batch unless empty
                    if not batch:
                        _jobs_cursor = _prefetch_jobs()
                        batch = _jobs_cursor.fetchmany(pending_limit - pending)
                        if not batch:
                            print()
                            self.__logger.debug(f'No computation can be booked for embedding #{embedding_id}.')

                # Mark the media_ids as taken, to avoid duplicate computation from other concurrent processes
                if stop.is_set():
                    batch = []
                elif batch:
                    _status('booking...')
                    booked += len(batch)
                    pending += len(batch)
                    _status(f'booked += {len(batch):d}')
                    print()
                    self.executemany(
                        "INSERT INTO ManifoldHoles (manifold_id, media_id, msg) VALUES (?, ?, '(COMPUTING)')",
                        ((manifold_id, mi) for mi, _ in batch))
                    # NOTE: in case of graceful interruption, these the placeholders will have to be filled.

                # End the transaction
                _status('committing...')
                self.commit()

                # Enqueue jobs, if any
                for _job in batch:
                    _status('enqueuing...')
                    enqueued += 1
                    actor_sys.tell('fetch', _job)

                # Prepare termination if stopping; terminate if no pending jobs
                if pending <= 0:
                    _status('terminating...')
                    assert actor_sys.pending == 0, actor_sys.status
                    assert pending == 0, (pending, enqueued, written, ignored, failures)

                    if notice is not None and not stop.is_set():
                        # Wait! Let's keep the loop alive... Just wait for an external event.
                        _status('waiting...')
                        print()
                        assert notice.wait()
                        notice.clear()
                    else:
                        print()
                        break

        if stop.is_set():
            self.__logger.warning(f'Embedding creation interrupted after {failures:d} failures.')
        else:
            _lvl = self.__logger.info if failures == 0 else self.__logger.warning
            _lvl(f'Embedding creation finished with {failures:d} failures.')

        if threading.main_thread() == threading.current_thread():
            # Could only call signal in main thread
            # noinspection PyUnboundLocalVariable
            signal.signal(signal.SIGINT, _previous_interrupt_handler)

        assert computed == index.get_n_items()
        return written

    def _build_manifold_index(self, manifold_id: int, n_trees: int, *, commit: bool = True) -> int:
        """Will commit multiple write transactions"""
        index = self._get_annoy_index(manifold_id)
        n_items = index.get_n_items()

        try:
            self.__logger.info(f'Building index forest of {n_trees} trees with {n_items} items...')
            index.build(n_trees)
        except KeyboardInterrupt:
            self.__logger.warning('Forest building interrupted.')
        else:
            got_n_trees = index.get_n_trees()
            # NOTE: the following UPDATE query was sometimes failing (because the db was locked)
            # => start a TX, but consider different method calls...
            if not self._db.in_transaction:
                self.begin_exclusive_transaction()
            # The above problem seems to be solved...
            metadata = self._get_manifold_metadata(manifold_id)
            metadata.update(dict(
                n_trees=got_n_trees,
                utc=str(datetime.utcnow()),
            ))
            self._db.execute('UPDATE Manifolds SET building = 0, ready = 1, metadata = ? '
                             'WHERE manifold_id = ?;', (json.dumps(metadata), manifold_id,))
            self.__logger.info(f'Ann index building finished; Manifold {manifold_id:d} marked as ready.')
            if commit:
                self._db.commit()
            self.__manifolds_annoy_index[manifold_id] = index

            return got_n_trees

    def reindex_manifold(self, manifold_id: int, metric: str = 'euclidean', n_trees: int = 10) -> int:

        old_index = self._get_annoy_index(manifold_id)

        with self._db:
            self.begin_exclusive_transaction()
            if not self._db.execute('SELECT ready FROM Manifolds WHERE manifold_id = ?', (manifold_id,)).fetchone()[0]:
                raise RuntimeError(f'Could not reindex manifold #{manifold_id} which is not ready.')

            full_fn = mkstemp(dir=self.get_data_dir(),
                              prefix=f'{manifold_id:06d}.',
                              suffix='.annoy', text=False)[1]

            index = AnnoyIndex(old_index.f, metric)
            index.on_disk_build(full_fn)
            self.__manifolds_annoy_index[manifold_id] = index

            fn = os.path.relpath(full_fn, self.get_data_dir())
            metadata = self._get_manifold_metadata(manifold_id)
            metadata.update(dict(
                fn=fn,
                metric=metric,
                utc=str(datetime.utcnow()),
            ))
            self.__logger.debug(f"Created new index on {fn}")

            self._db.execute(
                'UPDATE Manifolds SET building = 1, ready = 0, metadata = ? '
                'WHERE manifold_id = ?;', (json.dumps(metadata), manifold_id))

            self.commit()

        self.__logger.info("Copying items from old index...")
        for item_i in range(old_index.get_n_items()):
            index.add_item(item_i, old_index.get_item_vector(item_i))

        return self._build_manifold_index(manifold_id, n_trees=n_trees)

    def merge_manifolds(self, embedding_id: int, metric: str = 'euclidean', n_trees: int = 10) -> Tuple[int, int]:
        with self._db:
            # Create new manifold
            dim: int = self.execute('SELECT dim FROM Embeddings WHERE embedding_id = ?', (embedding_id,)).fetchone()[0]
            manifold_id = self._make_new_manifold(embedding_id=embedding_id, metric=metric, dim=dim)
            index = self._get_annoy_index(manifold_id)
            self.__logger.debug(f"Created new manifold #{manifold_id}")

            # Start a very long transaction (TODO: split the tx)
            self.begin_exclusive_transaction()

            # See which manifolds will be used
            c = self.execute(
                'SELECT manifold_id FROM Manifolds '
                'WHERE ready AND embedding_id = ? ',
                (embedding_id,))
            merged_manifold_ids = list(row[0] for row in c)
            self.__logger.debug(f"Copying items from {merged_manifold_ids=}...")

            # Copy items from ready manifolds
            merged_manifold_counter = Counter()
            c = self.execute(
                'WITH MergedItems AS ( '
                '   SELECT manifold_id, item_i, media_id '
                '   FROM ManifoldItems NATURAL JOIN Manifolds '
                '   WHERE ready AND embedding_id = ? '
                '   GROUP BY media_id '
                '   ORDER BY MIN(manifold_id)) '
                'SELECT manifold_id, item_i, media_id FROM MergedItems '
                'WHERE (manifold_id, item_i, media_id) IN ManifoldItems',  # This should be a redundant check
                (embedding_id,))

            def _gen_params():
                item_i = index.get_n_items()
                assert item_i == 0
                for old_manifold_id, old_item_i, media_id, in c:
                    merged_manifold_counter.update({old_manifold_id})
                    old_index = self._get_annoy_index(old_manifold_id)
                    try:
                        v = old_index.get_item_vector(old_item_i)
                    except Exception:
                        if old_item_i >= old_index.get_n_items():
                            self.__logger.warning(f"Index is short: {old_item_i=:d} >= {old_index.get_n_items()=:d}")
                        self.__logger.error(f"Could not read item #{old_item_i} from index "
                                            f"of manifold #{old_manifold_id}", exc_info=True)
                        self.execute('ROLLBACK TRANSACTION')
                        raise
                    index.add_item(item_i, v)
                    yield manifold_id, item_i, media_id
                    item_i += 1  # WARNING: do not increment before yielding the value associated with media_id ...

            c2 = self.executemany(
                'INSERT INTO ManifoldItems (manifold_id, item_i, media_id) VALUES (?, ?, ?)', _gen_params())
            inserted = c2.rowcount
            self.__logger.debug(f"Copied {inserted} items to manifold #{manifold_id}.")

            merged_manifold_counts = {mi: 0 for mi in merged_manifold_ids}
            merged_manifold_counts.update(dict(merged_manifold_counter))

            # Also copy holes that are not covered by the merge
            self.execute(
                'INSERT INTO ManifoldHoles '
                'SELECT :manifold_id, media_id, msg '
                'FROM ManifoldHoles NATURAL JOIN Manifolds '
                'WHERE ready AND embedding_id = :embedding_id AND manifold_id != :manifold_id '
                'AND media_id NOT IN (SELECT media_id FROM ManifoldItems WHERE manifold_id = :manifold_id) '
                'GROUP BY media_id',
                dict(embedding_id=embedding_id, manifold_id=manifold_id))

            # Update metadata to trace merged origin
            self.__logger.debug(f"Updating metadata...")
            metadata = self._get_manifold_metadata(manifold_id)
            metadata.update(dict(
                merged_from=merged_manifold_counts,
                utc=str(datetime.utcnow()),
            ))
            self.execute(
                'UPDATE Manifolds SET metadata = ? WHERE manifold_id = ?',
                (json.dumps(metadata), manifold_id))

            # Actually build the manifold index
            self._build_manifold_index(manifold_id, n_trees, commit=False)

            # Set old manifolds as not ready
            self.executemany(
                'UPDATE Manifolds SET ready = 0, merged = 1 WHERE manifold_id = ?',
                ((mi,) for mi in merged_manifold_ids))

            # Commit the very long transaction
            self.commit()

            return manifold_id, inserted

    def print_manifolds(self, *, w=90):
        self.print_embeddings(w=w)

        c = self.execute(
            'SELECT embedding_id, manifold_id, ready, building, merged, inactive, metadata '
            'FROM Manifolds ORDER BY embedding_id, manifold_id ')
        print()
        print()
        print('Manifolds (grouped by embedding):')
        print()
        print(" embedding_id ╷ manifold_id status       items    holes   metadata")
        # print('─' * 14 + '┼' + '─' * (w - 14 - 1))
        print('─' * w)
        prev_embedding_id = None
        for embedding_id, manifold_id, mr, mb, me, mi, mm in c:
            if embedding_id != prev_embedding_id:
                if prev_embedding_id is not None:
                    # print('─' * 14 + '┼' + '─' * (w - 14 - 1))
                    print('┄' * w)
                prev_embedding_id = embedding_id
            try:
                status: str = {
                    (True, False, False, False): '\033[92mREADY    \033[0m',
                    (False, True, False, False): '\033[96mBUILDING \033[0m',
                    (False, False, True, False): '\033[94mMERGED   \033[0m',
                    (False, False, False, True): '\033[34mINACTIVE \033[0m',
                }[mr, mb, me, mi]
            except KeyError:
                status: str = '\033[33mUNKNOWN  \033[0m'
            items: int = self.execute(
                'SELECT COUNT(*) FROM ManifoldItems WHERE manifold_id = ?', (manifold_id,)).fetchone()[0]
            holes: int = self.execute(
                'SELECT COUNT(*) FROM ManifoldHoles WHERE manifold_id = ?', (manifold_id,)).fetchone()[0]
            print(f" {embedding_id:12d} ┆ {manifold_id:11d} {status:s} {items:8d} {holes:8d}   {mm}")
        # print('─' * 14 + '┴' + '─' * (w - 14 - 1))
        print('─' * w)

    # TODO: add explicit command to forget item holes (so that it may later retry)

    # TODO: add command to remove manifolds that have lost their underlying annoy index (user deleted the file).


class _MaybeFetchActor(Actor):
    def __init__(self, db_data_dir):
        super().__init__()
        self.db = LocationsDatabase(data_dir=db_data_dir)

    def receive(self, msg: Any):
        media_id, url = msg
        try:
            data = None if url.startswith('file:') else self.db.fetch(url)
        except Exception as ex:
            _, __, tb = sys.exc_info()
            ex.with_traceback(tb)
            return 'main', (media_id, ex, None)
        else:
            return 'batch', (media_id, url, data)


class _EmbeddingBatchingActor(Actor):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def receive(self, msg: Any):
        id_batch, emb_batch = [], None
        _next_msgs = self.ask_available(block=True, timeout=.1)
        for _msg in itertools.chain([msg], _next_msgs):
            media_id, url, data = _msg
            try:
                mapped = self.embedding.batching_map(url=url, data=data)
                emb_batch = self.embedding.batching_reduce(batch=emb_batch, mapped=mapped)
            except Exception as ex:
                _, __, tb = sys.exc_info()
                ex.with_traceback(tb)
                self.tell('main', (media_id, ex, None))
            else:
                id_batch.append(media_id)
            if len(id_batch) >= self.embedding.batch_size:
                break
        if id_batch:
            return 'embed', (id_batch, emb_batch)


class _EmbeddingTransformActor(Actor):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def receive(self, msg: Any):
        id_batch, emb_batch = msg
        try:
            vv = self.embedding.batching_transform(batch=emb_batch)
        except Exception as ex:
            _, __, tb = sys.exc_info()
            ex.with_traceback(tb)
            for media_id in id_batch:
                self.tell('main', (media_id, ex, None))
        else:
            for media_id, v in zip(id_batch, vv):
                self.tell('annoy', (media_id, v))


class _AnnoyAddActor(Actor):
    def __init__(self, index: AnnoyIndex):
        super().__init__()
        self.index = index
        self.item_i = index.get_n_items()

    def receive(self, msg: Any):
        media_id, v = msg
        self.index.add_item(self.item_i, v)
        reply = 'main', (media_id, None, self.item_i)
        self.item_i += 1
        # assert self.index.get_n_items() == self.item_i
        return reply
