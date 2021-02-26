import os
import warnings
from tempfile import TemporaryDirectory
import numpy as np
import pytest

from aptl3.db import Database
from aptl3.scripts.load_coco import load_coco

coco_dir = './coco/annotations/'


@pytest.mark.skipif(not os.path.isdir(coco_dir), reason='test_coco would not find the coco annotations json.')
def test_coco():

    with TemporaryDirectory() as _data_dir:
        db = Database(data_dir=_data_dir)

        _train_images = 5  # This shall be a short test...

        load_coco(db=db, coco_dir=coco_dir, data_type='train2017', max_samples=_train_images)

        c = db.execute('SELECT relation_id from Relations')
        relation_id: int = c.fetchone()[0]

        c = db.execute("SELECT COUNT(*) FROM MediaLocations WHERE url NOT LIKE 'data:,%'")
        _inserted_images: int = c.fetchone()[0]
        assert _inserted_images == _train_images

        c = db.execute("SELECT COUNT(*) FROM MediaLocations WHERE url LIKE 'data:,%'")
        _inserted_sentences: int = c.fetchone()[0]
        assert _inserted_sentences >= _train_images

        embedding_id_images = db.add_embedding('image')
        embedding_id_sentences = db.add_embedding('sentence')

        manifold_id_images, _inserted = db.build_new_manifold(embedding_id=embedding_id_images)
        assert _inserted == _inserted_images + _inserted_sentences  # items + holes

        manifold_id_sentences, _inserted = db.build_new_manifold(embedding_id=embedding_id_sentences)
        assert _inserted == _inserted_sentences + _inserted_images  # items + holes

        gpa_id, gpa = db.build_generalized_procrustes(
            src_embedding_ids=(embedding_id_images, embedding_id_sentences),
            src_relation_ids=(relation_id,),
            min_samples=_inserted_images,
            max_samples=_inserted_sentences,
        )

        assert gpa.procrustes_distance < 1

        ################################################################################################################

        _val_images = 3
        load_coco(db=db, coco_dir=coco_dir, data_type='val2017', max_samples=_val_images)

        _, _inserted = db.build_new_manifold(embedding_id=embedding_id_images)
        assert _inserted >= _val_images
        _, _inserted = db.build_new_manifold(embedding_id=embedding_id_sentences)
        assert _inserted >= _val_images

        ################################################################################################################

        c = db.execute('SELECT media_id FROM Media')
        _tested = 0
        for media_id, in c:
            for _embedding_id, v in db.get_media_vectors(media_id=media_id):
                w = gpa.predict(src_embedding_id=_embedding_id, dest_embedding_id=_embedding_id, x=np.atleast_2d(v))[0]
                dist = np.linalg.norm(v-w)
                assert dist >= -1.0e-5
                assert dist < 0.2  #
                if dist > 0.01:
                    warnings.warn(f'{dist=:.4f} should be approximately 0 as we did v @ R @ R.T with R orthogonal.')
                _tested += 1

        assert _tested >= _train_images + _val_images
