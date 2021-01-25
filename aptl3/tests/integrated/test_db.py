import os
from tempfile import TemporaryDirectory
from urllib.request import pathname2url

from aptl3.db import Database
from aptl3.procrustes.generalized import GeneralizedProcrustesAnalysis


def test_db():
    with TemporaryDirectory() as _data_dir:
        db = Database(data_dir=_data_dir)
        assert db.get_data_dir() == _data_dir

        pathname: str = os.path.join(os.path.dirname(__file__), '../rose.jpg')
        assert os.path.isfile(pathname)
        size: int = os.path.getsize(pathname)
        url: str = 'file:' + pathname2url(pathname)

        data: bytes = db.fetch(url)
        assert len(data) == size

        media_id, _flags = db.ingest_url(url=url)
        assert _flags.first_access
        assert not _flags.failed
        assert not _flags.new_failure
        assert not _flags.changed

        ################################################################################################################

        _, _inserted = db.search(q=media_id.hex())
        assert _inserted == 0  # no embeddings => no results

        ################################################################################################################

        embedding_id: int = db.add_embedding('r-16')

        manifold_id, _inserted = db.build_new_manifold(embedding_id=embedding_id)
        assert _inserted == 1

        results_id, _inserted = db.search(q=media_id.hex())
        assert _inserted == 1  # 4 in the second case

        c = db.execute('SELECT * FROM results.ResultsMedia WHERE results_id = ?', (results_id,))
        rows = c.fetchall()
        assert len(rows) == 1
        _results_id, _media_id, _info, _rank = rows[0]
        assert _results_id == results_id
        assert _media_id == media_id
        assert 0 <= _rank

        ################################################################################################################

        embedding_id2: int = db.add_embedding('r-32')

        manifold_id2, _inserted = db.build_new_manifold(embedding_id=embedding_id2)
        assert _inserted == 1

        relation_id: int = db.create_relation(dict())

        db.add_media_relation(relation_id=relation_id, media_id=media_id, other_media_id=media_id)

        gpa_id, _ = db.build_generalized_procrustes(
            src_embedding_ids=(embedding_id, embedding_id2),
            src_relation_ids=(relation_id,),
            min_samples=1,
        )

        gpa = db.load_generalized_procrustes_analysis(gpa_id=gpa_id)
        assert isinstance(gpa, GeneralizedProcrustesAnalysis)
        assert set(gpa.orthogonal_models.keys()) == {embedding_id, embedding_id2}
        assert gpa.dim == 32  # max dimensionality of the two embeddings

        ################################################################################################################

        results_id, _inserted = db.search(q=media_id.hex())
        assert _inserted == 4  # r-16, r-32, r-16 gpa r-32, r-32 gpa r-16

        c = db.execute('SELECT * FROM results.ResultsMedia WHERE results_id = ?', (results_id,))
        rows = c.fetchall()
        assert len(rows) == 4
        for _results_id, _media_id, _info, _rank in rows:
            assert _results_id == results_id
            assert _media_id == media_id
            assert 0 <= _rank

        ################################################################################################################

        missing, inserted = db.try_load_missing_thumbs()
        assert missing == 0
        assert inserted == 1

        ################################################################################################################

        got_n_trees = db.reindex_manifold(manifold_id=manifold_id, n_trees=13)
        assert got_n_trees == 13

        got_n_trees2 = db.reindex_manifold(manifold_id=manifold_id2, n_trees=42)
        assert got_n_trees2 == 42

        ################################################################################################################

        manifold_id_empty, _inserted = db.build_new_manifold(embedding_id=embedding_id)
        assert _inserted == 0  # already inserted in old manifold!

        manifold_id_merged, _inserted = db.merge_manifolds(embedding_id=embedding_id)
        assert _inserted == 1
