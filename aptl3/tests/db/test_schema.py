import os
import random
import time
from tempfile import TemporaryDirectory
from threading import Thread, Barrier

from ...db.schema import SchemaDatabase


def test_db():
    assert isinstance(SchemaDatabase.get_default_data_dir(), str)
    with TemporaryDirectory() as _data_dir:
        db = SchemaDatabase(data_dir=_data_dir)
        assert db.get_data_dir() == _data_dir
        assert os.path.isfile(db.get_db_file())
        assert len(db.execute('SELECT * FROM Media').fetchall()) == 0


def test_db_exclusive_threads():
    tx_duration = .01
    free_sleep = .001
    n_th = 20
    n_tx = 10
    # expected minimum total duration = n_tx * n_th * tx_duration
    persistence = 5.

    with TemporaryDirectory() as _data_dir:
        SchemaDatabase(data_dir=_data_dir).execute(
            'PRAGMA busy_timeout = 1')  # Should have no effect.
        barrier = Barrier(n_th)

        def _job():
            barrier.wait()  # Let's spam sqlite by starting all at once
            db = SchemaDatabase(data_dir=_data_dir)
            barrier.wait()
            for _ in range(n_tx):
                poll = tx_duration * (1. + n_tx * n_th * random.random() / persistence)
                db.begin_exclusive_transaction(poll=poll)
                time.sleep(tx_duration)
                db.commit()
                time.sleep(free_sleep)

        threads = [Thread(target=_job, daemon=True) for _ in range(n_th)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()


if __name__ == '__main__':
    import logging
    logging.basicConfig(level='DEBUG',
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    test_db()
    test_db_exclusive_threads()
