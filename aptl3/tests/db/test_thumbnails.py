import os
from tempfile import TemporaryDirectory
from urllib.request import pathname2url

from ...db.thumbnails import ThumbnailsDatabase


def test_thumbs_load():
    with TemporaryDirectory() as _data_dir:
        db = ThumbnailsDatabase(data_dir=_data_dir)

        _url = 'file:' + pathname2url(os.path.join(os.path.dirname(__file__), '..', 'rose.jpg'))
        db.ingest_url(_url)

        assert db.thumbs_count_missing() == 1

        db.thumbs_load()

        assert 0 <= db.thumbs_process.pending <= 1, db.thumbs_process
        assert 0 <= db.thumbs_process.working <= 1, db.thumbs_process
        print(db.thumbs_process.status)

        db.thumbs_finish(block=True)
        print(db.thumbs_process.status)

        assert db.thumbs_process.pending == 0, (db.thumbs_process.working, db.thumbs_process.status)
        assert db.thumbs_process.working == 0, (db.thumbs_process.working, db.thumbs_process.status)
        assert db.thumbs_progress.computed == 1, db.thumbs_progress
        assert db.thumbs_progress.nulls == 0, db.thumbs_progress
        assert db.thumbs_count_missing() == 0, db.thumbs_count_missing()
        print(db.thumbs_process.status)


if __name__ == '__main__':
    test_thumbs_load()
