from tempfile import TemporaryDirectory

from ...db.manifolds import ManifoldsDatabase


def test_bg_manifold_build():
    with TemporaryDirectory() as _data_dir:
        db = ManifoldsDatabase(data_dir=_data_dir)

        embedding_id: int = db.add_embedding('r-16')

        db.ingest_url('data:,This is a test sentence')

        manifold_id, inserted = db.build_new_manifold(embedding_id=embedding_id)

        assert inserted == 1

        db.print_manifolds()
