import logging
import os
import warnings
from contextlib import nullcontext
from typing import NamedTuple, Optional, Tuple, Dict, Iterator
from urllib.request import urlopen, pathname2url, url2pathname

from requests_cache import CachedSession

from .schema import SchemaDatabase


class UrlIngestFlags(NamedTuple):
    first_access: bool
    failed: bool
    new_failure: bool
    changed: bool


class LocationsDatabase(SchemaDatabase):

    def __init__(self, data_dir: Optional[str] = None):
        super().__init__(data_dir)

        env_var = f'{__name__:s}.requests_cache'.replace('.', '_').upper()
        fn = os.environ.get(env_var, None)
        if fn is None:
            logging.debug(f'Environment variable {env_var} not found.')
            fn = os.path.join(self.get_data_dir(), 'requests_cache.sqlite')
        logging.warning(f'Using requests cache at {os.path.abspath(fn)}')

        fn = fn[:-len('.sqlite')] if fn.endswith('.sqlite') else fn
        self.__cached_session = CachedSession(cache_name=fn)
        self.__hash_constructors: Dict[str, callable] = dict()

    @staticmethod
    def _get_default_hash_function() -> str:
        return 'xxh3_64'

    def _get_hash(self, name: Optional[str] = None):
        if name is None:
            name = self._get_default_hash_function()
        try:
            _constructor = self.__hash_constructors[name]
        except KeyError:
            if name == 'md5':
                from hashlib import md5
                _constructor = md5
            elif name == 'xxh3_64':
                # http://cyan4973.github.io/xxHash/
                from xxhash import xxh3_64
                _constructor = xxh3_64
            elif name == 'xxh3_128':
                from xxhash import xxh3_128
                _constructor = xxh3_128
            # May add other hash functions here
            else:
                from hashlib import new
                return new(name)
            self.__hash_constructors[name] = _constructor
        return _constructor()

    def url_to_media_id(self, url: str, *, hash_function: Optional[str] = None) -> bytes:
        h = self._get_hash(name=hash_function)

        with self.open(url) as _f:
            while True:
                b = _f.read(4096)
                h.update(b)
                if len(b) <= 0:
                    break

        return h.digest()

    def open(self, url: str):
        """
        Please use as a context manager like `with db.open("http://...") as f: ...`
        """
        if url.startswith('http:') or url.startswith('https:') or url.startswith('ftp:'):
            with self.__cached_session.get(url) as resp:
                return nullcontext(resp.raw)
        if url.startswith('file:'):
            return open(url2pathname(url.lstrip('file:')), mode='rb')
        else:
            return urlopen(url)

    def fetch(self, url: str) -> bytes:
        with self.open(url) as f:
            return f.read(-1)

    # noinspection SqlResolve
    def __write_url_media_id(self, url: str, new_hash_function: str, new_media_id: Optional[bytes], *,
                             commit: bool = True, update_last_access: bool = True) -> UrlIngestFlags:
        """
        :param url:
        :param new_media_id: None means it failed
        :return:
        """
        with (self._db if commit else nullcontext()):
            c = self._db.cursor()

            c.execute('SELECT hash_function, media_id FROM MediaLocations WHERE url = ?', (url,))
            already_accessed, old_hash_function, old_media_id = False, None, None
            for old_hash_function, old_media_id in c:
                assert old_media_id is None or isinstance(old_media_id, bytes)
                already_accessed = True
            # NOTE: the above `for` is not really an iteration as url is unique and there is at most 1 associated media

            if new_media_id is None:
                if not already_accessed:
                    c.execute('INSERT INTO MediaLocations '
                              '(url, media_id, ts_first_access, ts_last_access, ts_first_fail) '
                              'VALUES (:url, NULL, datetime("now"), datetime("now"), datetime("now"))',
                              {'url': url})
                    return UrlIngestFlags(first_access=True, failed=True, new_failure=True, changed=False)
                elif old_media_id is None:
                    if update_last_access:
                        c.execute('UPDATE MediaLocations SET ts_last_access = datetime("now") '
                                  'WHERE url = :url', {'url': url})
                    return UrlIngestFlags(first_access=False, failed=True, new_failure=False, changed=False)
                else:
                    c.execute('UPDATE MediaLocations SET media_id = NULL, ts_last_access = datetime("now"), '
                              'ts_first_fail = datetime("now") WHERE url = :url', {'url': url})
                    return UrlIngestFlags(first_access=False, failed=True, new_failure=True, changed=False)
            else:
                if old_media_id != new_media_id:
                    c.execute('INSERT OR IGNORE INTO Media (media_id) VALUES (?)', (new_media_id,))

                if not already_accessed:
                    c.execute('INSERT INTO MediaLocations '
                              '(url, hash_function, media_id, ts_first_access, ts_last_access) '
                              'VALUES (:url, :hash_function, :new_media_id, datetime("now"), datetime("now"))',
                              {'url': url, 'hash_function': new_hash_function, 'new_media_id': new_media_id})
                    return UrlIngestFlags(first_access=True, failed=False, new_failure=False, changed=False)
                else:
                    if old_media_id != new_media_id or update_last_access:
                        if old_hash_function != new_hash_function:
                            new_media_id_old_hf = self.url_to_media_id(url=url, hash_function=old_hash_function)
                            media_changed = new_media_id_old_hf != old_media_id
                            if not media_changed:
                                warnings.warn(  # Just notifying that the default hash function may have changed.
                                    f'A rehashing operation has started '
                                    f'({old_hash_function=:s}, {new_hash_function=:s}).')
                                self._update_rehashed_media(
                                    old_media_id=old_media_id, new_media_id=new_media_id, commit=False)
                        else:
                            media_changed = old_media_id != new_media_id
                        c.execute('UPDATE MediaLocations SET '
                                  'hash_function = :new_hash_function, '
                                  'media_id = :new_media_id, '
                                  'ts_last_access = datetime("now") '
                                  ' WHERE url = :url',
                                  {'url': url, 'new_hash_function': new_hash_function, 'new_media_id': new_media_id})
                    else:
                        media_changed = False
                    return UrlIngestFlags(first_access=False, failed=False, new_failure=False, changed=media_changed)

    def try_ingest_url(self, url: str, *,
                       commit: bool = True, update_last_access: bool = True) -> Tuple[bytes, UrlIngestFlags, Exception]:
        """ Store failures and returns exceptions """
        # noinspection PyBroadException
        new_hash_function = self._get_default_hash_function()
        try:
            new_media_id = self.url_to_media_id(url, hash_function=new_hash_function)
            ex = None
        except Exception as _ex:
            ex = _ex
            new_media_id = None
        return new_media_id, self.__write_url_media_id(
            url, new_hash_function, new_media_id,
            commit=commit, update_last_access=update_last_access), ex

    def ingest_url(self, url: str, *,
                   commit: bool = True, update_last_access: bool = True) -> Tuple[bytes, UrlIngestFlags]:
        """ Does not store failures but raises exceptions """
        new_hash_function = self._get_default_hash_function()
        new_media_id = self.url_to_media_id(url, hash_function=new_hash_function)
        return new_media_id, self.__write_url_media_id(
            url, new_hash_function, new_media_id,
            commit=commit, update_last_access=update_last_access)

    def _update_rehashed_media(self, *, old_media_id: bytes, new_media_id: bytes, commit: bool = True):
        """Called when a media did not change in content but its media_id changed after changing the hash function."""
        # Will transfer old data / replace media_id in the following tables / rows:
        # - new Media row: copy parent_id, metadata from old row
        # - Thumbnails: replace media_id
        # - ManifoldItems: replace media_id
        # - ManifoldHoles: replace media_id
        # - MediaRelations: replace media_id
        # - MediaRelations: replace other_media_id
        # - old Media row: can be safely deleted as last operation
        # It is assumed that MediaLocations does not need to be updated by this method.
        with (self._db if commit else nullcontext()):
            c = self._db.cursor()
            c.execute(
                'UPDATE Media SET '
                'parent_id = (SELECT parent_id FROM Media WHERE media_id = :old_media_id), '
                'metadata = (SELECT metadata FROM Media WHERE media_id = :old_media_id) '
                'WHERE media_id = :new_media_id',
                {'old_media_id': old_media_id, 'new_media_id': new_media_id})
            c.execute(
                'UPDATE Thumbnails SET media_id = :new_media_id WHERE media_id = :old_media_id',
                {'old_media_id': old_media_id, 'new_media_id': new_media_id})
            c.execute(
                'UPDATE ManifoldItems SET media_id = :new_media_id WHERE media_id = :old_media_id',
                {'old_media_id': old_media_id, 'new_media_id': new_media_id})
            c.execute(
                'UPDATE ManifoldHoles SET media_id = :new_media_id WHERE media_id = :old_media_id',
                {'old_media_id': old_media_id, 'new_media_id': new_media_id})
            c.execute(
                'UPDATE MediaRelations SET media_id = :new_media_id WHERE media_id = :old_media_id',
                {'old_media_id': old_media_id, 'new_media_id': new_media_id})
            c.execute(
                'UPDATE MediaRelations SET other_media_id = :new_media_id WHERE other_media_id = :old_media_id',
                {'old_media_id': old_media_id, 'new_media_id': new_media_id})
            c.execute(
                'DELETE FROM Media WHERE media_id = :old_media_id',
                {'old_media_id': old_media_id})

    def ingest_file_directory(self, source_dir: str, *,
                              commit: bool = True, update_last_access: bool = True) -> dict:
        first_access: int = 0
        failed: int = 0
        new_failure: int = 0
        changed: int = 0

        with (self._db if commit else nullcontext()):
            for i, url in enumerate(walk_file_urls(source_dir=source_dir)):

                res = self.try_ingest_url(url=url, commit=False, update_last_access=update_last_access)
                media_id, flags, ex = res

                first_access += flags.first_access
                failed += flags.failed
                new_failure += flags.new_failure
                changed += flags.changed

                print(
                    f"scanned={i+1} "
                    f"{first_access=} "
                    f"{failed=} "
                    f"{new_failure=} "
                    f"{changed=} ",
                    end="\r", flush=True)
        print()

        return dict(
            scanned=i+1,
            first_access=first_access,
            failed=failed,
            new_failure=new_failure,
            changed=changed,
        )


def walk_file_urls(source_dir: str) -> Iterator[str]:
    source_dir = os.path.abspath(source_dir)
    if not os.path.isdir(source_dir):
        raise ValueError(source_dir)  # Not a directory
    sep = os.path.sep
    for root, dirs, files in os.walk(source_dir):
        for fname in files:
            fpath = root + sep + fname
            # assert os.path.isfile(fpath)
            yield 'file:' + pathname2url(fpath)
