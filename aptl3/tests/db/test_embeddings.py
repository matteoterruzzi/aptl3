from tempfile import TemporaryDirectory

from ...db.embeddings import EmbeddingsDatabase, Embedding


def test_bg_manifold_build():
    with TemporaryDirectory() as _data_dir:
        db = EmbeddingsDatabase(data_dir=_data_dir)

        embedding_name: str = 'r-16'

        embedding_id: int = db.add_embedding(embedding_name)
        assert embedding_id in list(db.list_ready_embedding_ids())
        assert embedding_id == db.get_embedding_id(embedding_name)

        new_embedding_id: int = db.push_new_embedding_version(embedding_name)
        assert new_embedding_id != embedding_id
        assert new_embedding_id in list(db.list_ready_embedding_ids())
        assert embedding_id not in list(db.list_ready_embedding_ids())
        assert new_embedding_id == db.get_embedding_id(embedding_name)

        assert isinstance(db.get_embedding(new_embedding_id), Embedding)

        other_embedding_id: int = db.get_embedding_id('r-32')

        url = 'data:,This is a test sentence'
        emb_vectors = list(db.get_url_vectors(url=url))
        assert len(emb_vectors) == 2
        assert all(embedding_id in [new_embedding_id, other_embedding_id] for embedding_id, v in emb_vectors)

        db.print_embeddings()

        del db
        # Let's reconnect.
        db = EmbeddingsDatabase(data_dir=_data_dir)

        assert other_embedding_id == db.get_embedding_id('r-32')
