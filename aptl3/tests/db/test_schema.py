import os
from tempfile import TemporaryDirectory
from ...db.schema import SchemaDatabase


def test_db():
    with TemporaryDirectory() as _data_dir:
        db = SchemaDatabase(data_dir=_data_dir)
        assert db.get_data_dir() == _data_dir
        assert os.path.isfile(db.get_db_file())
        assert len(db.execute('SELECT * FROM Media').fetchall()) == 0
