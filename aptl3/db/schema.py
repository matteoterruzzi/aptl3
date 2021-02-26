import os.path
import sqlite3
import logging
import sys
import threading
from time import perf_counter, sleep
from typing import Optional, Iterable, Callable


class SchemaDatabase:
    @staticmethod
    def get_default_data_dir():
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        return os.path.join(dname, '..', '..', 'data')
        # TODO: change this default using `appdirs.user_data_dir('aptl3', 'SomeAuthor')`

    def __init__(self, data_dir: Optional[str] = None):
        _create = False
        if data_dir is None or data_dir == '':
            data_dir = self.get_default_data_dir()
            _create = True
        data_dir = os.path.abspath(data_dir)
        self.__data_dir = data_dir
        fn = os.path.join(data_dir, 'db.sqlite3')
        if not os.path.exists(data_dir):
            if _create:
                logging.info(f"Working on new {data_dir=:s}")
                os.mkdir(data_dir)
            else:
                raise FileNotFoundError(data_dir)  # The specified directory does not exist.
        else:
            logging.info(f"Working on {data_dir=:s}")
        self._audit_open_files_allowed = set()
        sys.addaudithook(self.__audit_hook)
        is_new = not os.path.exists(fn)
        self._db: sqlite3.Connection = sqlite3.connect(
            'file:' + fn, uri=True, isolation_level='EXCLUSIVE')  # uri=True is for ATTACH commands.
        self.__db_file = fn
        if is_new:
            logging.debug(f"New db file, will create schema.")
            self.__execute_schema_script()

    def __audit_hook(self, name: str, args):
        if name == 'open':
            file, mode, flags = args
            if mode is not None and ('w' in mode or '+' in mode):
                if file in self._audit_open_files_allowed:
                    return
                if isinstance(file, int):
                    logging.debug(
                        f'Allowing audited open of numbered file {file:d}: '
                        f'{name}({", ".join(map(repr, args))})')
                    return
                if '.torch'+os.path.sep in file or os.path.sep+'torch'+os.path.sep in file:
                    # NOTE: sys.addaudithook could not be undone and the test suite would break without this rule.
                    logging.warning(
                        f'Allowing audited open of file with `torch/` in the path: '
                        f'{name}({", ".join(map(repr, args))})')
                    return
                if not os.path.abspath(file).startswith(self.__data_dir):
                    if f'{os.path.sep}.pytest_cache{os.path.sep}' in os.path.abspath(file):
                        logging.warning(
                            f'Allowing audited open of .pytest_cache file: '
                            f'{name}({", ".join(map(repr, args))})')
                        return
                    raise RuntimeError(
                        f'Denying audited open with write mode on file outside data dir: '
                        f'{name}({", ".join(map(repr, args))})')

    def get_db_file(self) -> str:
        return self.__db_file

    def get_data_dir(self) -> str:
        return self.__data_dir

    def __execute_schema_script(self):
        # This should be executed inside a transaction
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        fn = os.path.join(dname, 'schema.sql')
        with open(fn, 'r') as f:
            with self._db:
                self.begin_exclusive_transaction()
                c = self._db.cursor()
                c.executescript(f.read())
                self._db.commit()

    def begin_exclusive_transaction(self, poll: float = 1., timeout: float = 30., limit: int = 100,
                                    time_func: Callable[[], float] = perf_counter) -> float:
        """
        Begin a transaction acquiring an exclusive lock on the database.

        If `PRAGMA busy_timeout` is set, the value is temporarily replaced.

        The underlying db may in some cases ignore the busy timeout (e.g. after
        detecting a deadlock); in these cases the attempts limit will be reached.

        :param poll: busy timeout to wait before doing a new attempt
        :param timeout: Maximum total timeout before raising an OperationalError
        :param limit: Maximum number of attempts before raising an OperationalError
        :param time_func: Function to measure the time (defaults to time.perf_counter)
        :return:
        """
        start_time = time_func()
        assert not self._db.in_transaction
        _busy_timeout = int(self._db.execute('PRAGMA busy_timeout;').fetchone()[0])
        self._db.execute('PRAGMA busy_timeout=0;')
        while True:
            try:
                self._db.execute('BEGIN EXCLUSIVE TRANSACTION')
            except sqlite3.OperationalError as ex:
                logging.debug(f'{threading.current_thread().name} '
                              f'{self.__class__.__name__} '
                              f'Polling after {ex.__class__.__name__}: {str(ex):s} ... ')
                if int(1000 * poll) < 1:
                    sleep(poll)
                else:
                    self._db.execute(f'PRAGMA busy_timeout={int(1000 * poll):d};')
                limit -= 1
                if limit < 0 or time_func() - start_time > timeout:
                    raise
            else:
                self._db.execute(f'PRAGMA busy_timeout={_busy_timeout:d};')
                return time_func() - start_time

    def execute(self, sql: str, parameters: Iterable = ()) -> sqlite3.Cursor:
        return self._db.execute(sql, parameters)

    def executemany(self, sql: str, params) -> sqlite3.Cursor:
        return self._db.executemany(sql, params)

    def commit(self):
        self._db.commit()
