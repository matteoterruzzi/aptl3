from tempfile import TemporaryDirectory

from ...db.locations import LocationsDatabase


def test_bg_manifold_build():
    with TemporaryDirectory() as _data_dir:
        db = LocationsDatabase(data_dir=_data_dir)

        media_id, flags = db.ingest_url('data:,This is a test sentence')
        assert flags.first_access
        assert not flags.failed
        assert not flags.new_failure
        assert not flags.changed

        media_id, flags = db.ingest_url('data:,This is a test sentence')
        assert not flags.first_access
        assert not flags.failed
        assert not flags.new_failure
        assert not flags.changed

        media_id, flags = db.ingest_url('data:,This is a test sentence.')
        assert flags.first_access
        assert not flags.failed
        assert not flags.new_failure
        assert not flags.changed

        media_id, flags, ex = db.try_ingest_url('invalid url')
        assert media_id is None
        assert flags.first_access
        assert flags.failed
        assert flags.new_failure
        assert not flags.changed
        assert ex is not None

        media_id, flags, ex = db.try_ingest_url('invalid url')
        assert media_id is None
        assert not flags.first_access
        assert flags.failed
        assert not flags.new_failure
        assert not flags.changed
        assert ex is not None
