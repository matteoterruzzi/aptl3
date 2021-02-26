from tempfile import TemporaryDirectory
from time import sleep

from ...db.manifolds import ManifoldsDatabase


def test_bg_manifold_build():
    with TemporaryDirectory() as _data_dir:
        db = ManifoldsDatabase(data_dir=_data_dir)

        embedding_id: int = db.add_embedding('r-16')

        db.ingest_url('data:,This is a test sentence')
        db.notify_bg_manifold_build()

        db.ingest_url('data:,This is another test sentence')
        db.notify_bg_manifold_build()

        sleep(.2)  # Give time to let the bg thread terminate if the wait mechanism is broken

        db.ingest_url('data:,This is yet another test sentence')
        db.notify_bg_manifold_build()

        sleep(0)  # Still we don't guarantee termination of the background process with very short sleep before close.
        db.close_bg_manifold_builds()

        manifold_id: int = 1  # NOTE: weakly supported assumption
        assert db._get_manifold_embedding_id(manifold_id) == embedding_id

        index = db._get_annoy_index(manifold_id)
        assert 2 <= index.get_n_items() <= 3  # The third item may have not been booked in time before closing.
