import itertools
import json
import logging
import os
import signal
import threading
from collections import Counter
from datetime import datetime
from queue import Queue, Empty
from tempfile import mkstemp
from time import perf_counter
from typing import Tuple, Optional, Iterable, Dict, Any

from annoy import AnnoyIndex

from .embeddings import EmbeddingsDatabase
from .locations import LocationsDatabase
from ..embedding.abc import RequestIgnored


class ManifoldsDatabase(LocationsDatabase, EmbeddingsDatabase):
    """
    Storage and ANN indexing of high-dimensional vector data ("manifold items")
    abstracting away the index implementation and the possible multitude of indexes for one given embedding model.
    """

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self.__logger = logging.getLogger('manifolds')
        self.__manifolds_embedding: Dict[int, int] = dict()
        self.__manifolds_annoy_index: Dict[int, AnnoyIndex] = dict()

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

    def build_new_manifold(self, *, embedding_id: int, metric: str = 'euclidean', n_trees: int = 10) -> Tuple[int, int]:
        """Will commit multiple write transactions"""
        with self._db:
            manifold_id = self._make_new_manifold(embedding_id=embedding_id, metric=metric)
            inserted = self._populate_manifold(manifold_id=manifold_id)
            self._build_manifold_index(manifold_id, n_trees)
            return manifold_id, inserted

    def _populate_manifold(self, *, manifold_id: int) -> int:
        """Will commit multiple write transactions"""
        db = self
        embedding_id = self._get_manifold_embedding_id(manifold_id)
        embedding = self.get_embedding(embedding_id)
        index = self._get_annoy_index(manifold_id)

        stop = threading.Event()
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

        _embedding_input_queue = _StoppableQueue(maxsize=qsize*2)  # NOTE: all maxsizes are unreachable.
        _annoy_input_queue = _StoppableQueue(maxsize=qsize*2)
        _output_queue = Queue(maxsize=qsize*2)

        # NOTE: You may notice the system not enqueuing enough jobs to exploit the maximum concurrency.
        #       This can happen after the initial wave of jobs go through the fastest station.
        #       This queue system is designed to be steady and avoids accumulating jobs at the slowest station.

        def _status(_s: str):
            assert collected == computed + failures + ignored, (
                f"{collected=} {computed=} {failures=} {ignored=}"
            )
            _in_flight_to_be_written = collected - written
            _in_flight_to_be_enqueued = booked - enqueued
            _pending_in_other_threads = pending - _in_flight_to_be_written - _in_flight_to_be_enqueued
            assert (pending == booked - written
                    and 0 <= pending <= pending_limit
                    and 0 <= _in_flight_to_be_written <= pending_limit
                    and 0 <= _in_flight_to_be_enqueued <= pending_limit
                    and 0 <= _pending_in_other_threads <= pending_limit), (
                f"{pending_limit=} {pending=} {booked=} {enqueued=} {collected=} {written=} "
                f"{_in_flight_to_be_enqueued=} {_in_flight_to_be_written=} {_pending_in_other_threads=}")
            _queue_sizes = _embedding_input_queue.qsize(), _annoy_input_queue.qsize(), _output_queue.qsize()
            _working_in_other_threads = _pending_in_other_threads - sum(_queue_sizes)
            assert 0 <= _working_in_other_threads <= pending_limit, (
                f"{pending_limit=} {_working_in_other_threads=} {_pending_in_other_threads=} {_queue_sizes=}")
            throughput = computed / (perf_counter() - start_time)
            print(
                f"\r{_s:20s} {throughput=:5.2f}/s {computed=:d} {failures=:d} {ignored=:d} {pending=:3d} "
                f"queues: {_queue_sizes} concurrently working: {_working_in_other_threads}   ",
                end="\r", flush=True)

        # noinspection PyUnusedLocal
        def _interrupt_handler(signum, frame):
            stop.set()
            print(f'\n  >>> SIGINT Stopping as soon as possible... ')

        def _embedding_job():
            # TODO: replace with out-of-order batching executor in embedding module
            while True:
                try:
                    url, media_id = _embedding_input_queue.get()
                except _StoppableQueue.Stopped:
                    return
                try:
                    # NOTE: this is optimized for cached remote resources.
                    # FIXME: don't fetch data of local files (remember image files will not always be fully read).
                    v = embedding.transform(url=url, data=self.fetch(url))
                except Exception as ex:
                    if not isinstance(ex, RequestIgnored):
                        self.__logger.debug('Exception in embedding job:', exc_info=True)
                    _output_queue.put((ex, None, media_id))
                else:
                    _annoy_input_queue.put((v, media_id))
                _embedding_input_queue.task_done()

        def _annoy_job():
            # NOTE: This has a very bad performance when running the image embedding model.
            #       And it really is the index.add_item operation taking ~500ms.
            #       It becomes the slowest worker, much slower than the image embedding,
            #       but it goes back to ~normal when that model has terminated.
            #       No problem instead with the sentence embedding model.
            #       The problem must be caused by the threads badly concurring for something (CPU? I/O? what?)
            #       TODO: switch to multiprocessing instead of threading, just to try better understanding the issue;
            item_i = index.get_n_items()
            while True:
                try:
                    v, media_id = _annoy_input_queue.get()
                except _StoppableQueue.Stopped:
                    return
                index.add_item(item_i, v)
                _output_queue.put((None, item_i, media_id))
                item_i += 1
                _annoy_input_queue.task_done()

        def _prefetch_jobs():
            return self.execute(
                'SELECT url, media_id '
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

        def _main_job():
            nonlocal start_time, booked, enqueued, collected, written, computed, failures, ignored, pending

            _jobs_cursor = _prefetch_jobs()

            while True:
                items = []
                del_holes = []
                upd_holes = []

                # Collect previous results
                while enqueued - collected > 0:
                    _status('collecting...')
                    try:
                        ex, item_i, media_id = _output_queue.get_nowait()
                    except Empty:
                        if collected - written < write_minimum:
                            _status('processing...')
                            ex, item_i, media_id = _output_queue.get()
                        else:
                            break
                    collected += 1
                    if ex is None:
                        computed += 1
                        items += [(manifold_id, item_i, media_id)]
                        del_holes += [(manifold_id, media_id)]
                    else:
                        if isinstance(ex, RequestIgnored):
                            ignored += 1
                        else:
                            failures += 1
                            self.__logger.error(f'{type(ex).__name__}: {str(ex)}   ')
                        msg = f'{type(ex).__name__}: {str(ex)}'
                        upd_holes += [(msg, manifold_id, media_id)]
                    _output_queue.task_done()

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
                    for _i, (_, __media_id_) in reversed(list(enumerate(batch))):
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
                        ((manifold_id, mi) for _, mi in batch))
                    # NOTE: in case of graceful interruption, these the placeholders will have to be filled.

                # End the transaction
                _status('committing...')
                self.commit()

                # Enqueue jobs, if any
                for _job in batch:
                    _status('enqueuing...')
                    enqueued += 1
                    _embedding_input_queue.put(_job)

                # Prepare termination if stopping; terminate if no pending jobs
                if pending <= 0:
                    _status('terminating...')
                    _embedding_input_queue.join()
                    _annoy_input_queue.join()
                    assert _output_queue.empty()
                    assert pending == 0, (pending, enqueued, written, ignored, failures)
                    print()
                    self.__logger.info(f'Embedding #{embedding_id} is computed for all known media locations.')
                    break

        # Actually start working
        _embedding_th = threading.Thread(target=_embedding_job)
        _annoy_th = threading.Thread(target=_annoy_job)
        _embedding_th.start()
        _annoy_th.start()

        self.__logger.info(f'Populating index for manifold {manifold_id}...')
        _previous_interrupt_handler = signal.signal(signal.SIGINT, _interrupt_handler)
        _main_job()
        _embedding_input_queue.stop()
        _annoy_input_queue.stop()
        _embedding_th.join()
        _annoy_th.join()
        if stop.is_set():
            self.__logger.warning(f'Embedding creation interrupted after {failures:d} failures.')
        else:
            _lvl = self.__logger.info if failures == 0 else self.__logger.warning
            _lvl(f'Embedding creation finished with {failures:d} failures.')
        signal.signal(signal.SIGINT, _previous_interrupt_handler)
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
            # TODO: FIXME: the following UPDATE query is sometimes failing (because the db is locked)
            #  => maybe start a TX, but consider different method calls...
            if not self._db.in_transaction:
                self.begin_exclusive_transaction()
            # Question: is the above problem solved?
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


class _StoppableQueue(Queue):
    """
    Extends threading.Queue by adding two methods:
    - stop: for the controller to stop the producers and consumers
    - wait_for_stop: for producer and consumers to wait for a stop signal (with timeout)
    - get: for the consumers to get an item (blocking) or be notified to stop
    """
    class Stopped(Exception):
        pass

    def __init__(self, maxsize=0):
        super().__init__(maxsize)
        self.not_stopped = threading.Condition(self.mutex)
        self.stopped = False

    def stop(self):
        """Stops producers and consumers"""
        with self.mutex:
            self.stopped = True
            self.not_stopped.notify_all()
            self.not_empty.notify_all()

    def get(self, block=True, timeout=None):
        """Returns an item or raises StoppableQueue.Stopped"""
        if not block or timeout is not None:
            raise NotImplementedError
        with self.mutex:
            while not self._qsize():
                self.not_empty.wait()
                if self.stopped:
                    raise _StoppableQueue.Stopped()
            item = self._get()
            self.not_full.notify()
            return item
