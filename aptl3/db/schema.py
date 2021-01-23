import os.path
import sqlite3
import logging
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
        if data_dir is None or data_dir == '':
            data_dir = self.get_default_data_dir()
        data_dir = os.path.abspath(data_dir)
        self.__data_dir = data_dir
        fn = os.path.join(data_dir, 'db.sqlite3')
        if not os.path.exists(data_dir):
            logging.info(f"Working on new {data_dir=:s}")
            os.mkdir(data_dir)
        else:
            logging.info(f"Working on {data_dir=:s}")
        is_new = not os.path.exists(fn)
        self._db: sqlite3.Connection = sqlite3.connect(
            'file:' + fn, uri=True, isolation_level='EXCLUSIVE')  # uri=True is for ATTACH commands.
        self.__db_file = fn
        if is_new:
            logging.debug(f"New db file, will create schema.")
            self.__execute_schema_script()
        self.__attach_results_db()

    def get_db_file(self) -> str:
        return self.__db_file

    def get_data_dir(self) -> str:
        return self.__data_dir

    @staticmethod
    def get_attach_results_db_query() -> str:
        q = "ATTACH DATABASE 'file:results?mode=memory&cache=shared' AS results;"
        return q

    @staticmethod
    def get_schema_results_db_script() -> str:
        return '''
            CREATE TABLE IF NOT EXISTS results.Results (
                results_id INTEGER PRIMARY KEY AUTOINCREMENT,
                metadata TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS results.ResultsMedia (
                results_id INTEGER NOT NULL REFERENCES Results ON DELETE CASCADE,
                media_id BLOB NOT NULL REFERENCES Media ON DELETE CASCADE,
                info TEXT NULL,
                rank REAL NULL
            );
        '''

    def __attach_results_db(self):
        with self._db:
            self._db.executescript(self.get_attach_results_db_query())
            self._db.executescript(self.get_schema_results_db_script())

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

    def begin_exclusive_transaction(self, time_func: Callable[[], float] = None) -> float:
        if time_func is None:
            time_func = perf_counter
        start_time = time_func()
        while True:
            try:
                self.execute('BEGIN EXCLUSIVE TRANSACTION')
            except sqlite3.OperationalError as ex:
                print()
                logging.debug(f'Polling: {str(ex):s} ...')
                sleep(1.)
            else:
                return time_func() - start_time

    def execute(self, sql: str, parameters: Iterable = ()) -> sqlite3.Cursor:
        return self._db.execute(sql, parameters)

    def executemany(self, sql: str, params) -> sqlite3.Cursor:
        return self._db.executemany(sql, params)

    def commit(self):
        self._db.commit()
