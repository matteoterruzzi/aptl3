import logging
import sys
import traceback
from typing import Tuple, Optional, Any, NamedTuple

from .locations import LocationsDatabase
from ..am import ActorSystem, MapActor, Actor
from ..images import getSmallImage, imageToBytesIO


class ThumbsProgress(NamedTuple):
    computed: int
    nulls: int


class _MaybeFetchActor(Actor):
    def __init__(self, db_data_dir):
        super().__init__()
        self.db = LocationsDatabase(data_dir=db_data_dir)

    def receive(self, msg: Any):
        media_id, url = msg
        if url.startswith('data:,'):
            return 'main', (media_id, None, None)
        try:
            data = None if url.startswith('file:') else self.db.fetch(url)
        except Exception as ex:
            _, __, tb = sys.exc_info()
            ex.with_traceback(tb)
            return 'main', (media_id, ex, None)
        else:
            return 'thumb', (media_id, url, data)


def _url_to_thumb_bytes(msg: Any) -> Tuple[bytes, None, Optional[bytes]]:
    media_id, url, data = msg
    fp = data if data is not None else url
    # noinspection PyBroadException
    try:
        im = getSmallImage(fp)
        thumb = imageToBytesIO(im)
    except Exception:
        logging.debug(f'Skipping {url}', exc_info=True)
        return media_id, None, None
    else:
        return media_id, None, thumb.getvalue()


class ThumbnailsDatabase(LocationsDatabase):

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)
        self.__logger = logging.getLogger(__name__)
        self.__main_actor: Optional[_MainThumbLoadActor] = None
        self.__actor_system: Optional[ActorSystem] = None

    @property
    def thumbs_process(self) -> Optional[ActorSystem]:
        return self.__actor_system

    @property
    def thumbs_progress(self) -> Optional[ThumbsProgress]:
        if self.__main_actor is None:
            return None
        return ThumbsProgress(
            computed=self.__main_actor.computed,
            nulls=self.__main_actor.nulls,
        )

    def thumbs_load(self):
        """Wakeup background thumbnails loading (or start it if necessary)."""
        if self.__actor_system is None:
            self.__start_actor_system()
        self.__actor_system.tell('main', None)

    def thumbs_finish(self, block: bool = False):
        """Gracefully ask to finish the background thumbnails loading, if started."""
        if self.__main_actor and self.__actor_system:
            self.__actor_system.finish()
            if block:
                self.__actor_system.join_pending()

    def thumbs_select_missing(self, limit: int):
        """
        :param limit: number of results
        :return: database cursor (iterable of pairs of media_id, url)
        """
        return self.execute(
            'SELECT media_id, url FROM MediaLocations '
            'WHERE media_id not in (select media_id from Thumbnails) '
            'AND url NOT LIKE \'data:,%\' '
            'AND media_id is not NULL ORDER BY ts_last_access DESC LIMIT ?',
            (limit,))

    def thumbs_count_missing(self) -> int:
        """:return: number of media locations (other than "data:," urls) for which there's no thumbnail"""
        c = self.execute(
            'SELECT COUNT(*) FROM MediaLocations '
            'WHERE media_id not in (select media_id from Thumbnails) '
            'AND url NOT LIKE \'data:,%\' '
            'AND media_id is not NULL')
        return int(c.fetchone()[0])

    # TODO: Add method to forget failed

    def __start_actor_system(self, qsize: int = 512):

        _main_actor = _MainThumbLoadActor(self.get_data_dir(), qsize=qsize)

        # NOTE: the thumbnail extractor is very strange.
        # This will be faster with more processes than the number of CPUs
        s = ActorSystem(maxsize=qsize * 8, use_multiprocessing=True, monitor_thread=True)
        s.add_process('fetch', _MaybeFetchActor(self.get_data_dir()), pool=0)
        s.add_process('thumb', MapActor(_url_to_thumb_bytes, 'main'), pool=0)
        s.add_thread('main', _main_actor)

        s.start()
        self.__main_actor, self.__actor_system = _main_actor, s


class _MainThumbLoadActor(Actor):
    def __init__(self, data_dir: str, qsize: int = 512):
        super().__init__()
        self.__logger = logging.getLogger(__name__)
        self._data_dir = data_dir

        assert qsize >= 8
        self.book_minimum = qsize // 4
        self.pending_limit = qsize
        self.write_minimum = qsize // 8
        self.write_timeout = 30  # after which ignore write_minimum

        self.nulls = 0
        self.computed = 0
        self.collected = 0
        self.pending = 0

        self._db: Optional[ThumbnailsDatabase] = None
        self._jobs_cursor: Optional = None

    def init(self) -> None:
        self._db = ThumbnailsDatabase(data_dir=self._data_dir)
        self._jobs_cursor = self._db.thumbs_select_missing(self.pending_limit * 8)

    def receive(self, msg: Any) -> None:
        def _available_msgs():
            yield msg
            n = 1
            for _msg in self.ask_available(None, block=not self.finish_requested, timeout=self.write_timeout):
                if _msg is not None:
                    yield _msg
                    n += 1
                    if n >= self.write_minimum:
                        break
                if self.pending - n <= 0:
                    break
            for _msg in self.ask_available(None, block=False):
                if _msg is not None:
                    yield _msg
                    n += 1

        available_msgs = [] if msg is None else _available_msgs()
        thumbs = []

        for media_id, ex, thumb in available_msgs:
            if ex is not None:
                __err = f'{type(ex).__name__}: {str(ex)}'
                tb = traceback.TracebackException.from_exception(ex)
                print()
                self.__logger.error(__err)
                self.__logger.debug(''.join(tb.format()))
            if thumb is None:
                self.nulls += 1
            else:
                assert isinstance(thumb, bytes)
                assert len(thumb) > 100
                self.computed += 1
            self.collected += 1
            thumbs.append((media_id, thumb))

        self._db.begin_exclusive_transaction()

        self._db.executemany(
            'INSERT OR IGNORE INTO Thumbnails (media_id, thumbnail) '
            'VALUES (?, ?)', thumbs)
        self.pending -= len(thumbs)

        if (self.finish_requested and msg is not None
                or self.pending + self.book_minimum > self.pending_limit):
            batch = []
        else:
            batch = self._jobs_cursor.fetchmany(self.pending_limit - self.pending)
            if not batch:
                self._jobs_cursor = self._db.thumbs_select_missing(self.pending_limit * 8)
                batch = self._jobs_cursor.fetchmany(self.pending_limit - self.pending)
                if not batch:
                    self.__logger.debug(f'No thumbnails are missing.  ')

        if self.finish_requested and msg is not None:
            batch = []
        elif batch:
            self.pending += len(batch)

        self._db.commit()

        for _job in batch:
            self.tell('fetch', _job)
