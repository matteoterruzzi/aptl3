import os.path
from contextlib import nullcontext
from hashlib import md5
from typing import NamedTuple, Optional, Tuple
from urllib.request import urlopen

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

        fn = os.path.join(self.get_data_dir(), 'requests_cache')
        self.__cached_session = CachedSession(cache_name=fn)

    def url_to_media_id(self, url: str) -> bytes:
        h = md5()

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
        else:
            return urlopen(url)

    def fetch(self, url: str) -> bytes:
        with self.open(url) as f:
            return f.read(-1)

    # noinspection SqlResolve
    def __write_url_media_id(self, url: str, new_media_id: Optional[bytes], *,
                             commit: bool = True, update_last_access: bool = True) -> UrlIngestFlags:
        """
        :param url:
        :param new_media_id: None means it failed
        :return:
        """
        with (self._db if commit else nullcontext()):
            c = self._db.cursor()

            c.execute('SELECT media_id FROM MediaLocations WHERE url = ?', (url,))
            already_accessed, old_media_id = False, None
            for old_media_id, in c:
                assert old_media_id is None or isinstance(old_media_id, bytes)
                already_accessed = True

            if new_media_id is None:
                if not already_accessed:
                    c.execute('INSERT INTO MediaLocations '
                              '(url, media_id, ts_first_access, ts_last_access, ts_first_fail) '
                              'VALUES (:url, NULL, datetime("now"), datetime("now"), datetime("now"))',
                              {'url': url})
                    return UrlIngestFlags(first_access=True, failed=True, new_failure=True, changed=False)
                elif old_media_id is None:
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
                    c.execute('INSERT INTO MediaLocations (url, media_id, ts_first_access, ts_last_access) '
                              'VALUES (:url, :new_media_id, datetime("now"), datetime("now"))',
                              {'url': url, 'new_media_id': new_media_id})
                    return UrlIngestFlags(first_access=True, failed=False, new_failure=False, changed=False)
                else:
                    if old_media_id != new_media_id or update_last_access:
                        c.execute('UPDATE MediaLocations SET media_id = :new_media_id, ts_last_access = datetime("now")'
                                  ' WHERE url = :url', {'url': url, 'new_media_id': new_media_id})
                    return UrlIngestFlags(first_access=False, failed=False, new_failure=False,
                                          changed=(old_media_id != new_media_id))

    def try_ingest_url(self, url: str, *,
                       commit: bool = True, update_last_access: bool = True) -> Tuple[bytes, UrlIngestFlags, Exception]:
        """ Store failures and returns exceptions """
        # noinspection PyBroadException
        try:
            new_media_id = self.url_to_media_id(url)
            ex = None
        except Exception as ex:
            new_media_id = None
        return new_media_id, self.__write_url_media_id(
            url, new_media_id, commit=commit, update_last_access=update_last_access), ex

    def ingest_url(self, url: str, *,
                   commit: bool = True, update_last_access: bool = True) -> Tuple[bytes, UrlIngestFlags]:
        """ Does not store failures but raises exceptions """
        new_media_id = self.url_to_media_id(url)
        return new_media_id, self.__write_url_media_id(
            url, new_media_id, commit=commit, update_last_access=update_last_access)
