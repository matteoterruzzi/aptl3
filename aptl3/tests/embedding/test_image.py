import os
import numpy as np
from urllib.request import pathname2url

from ...embedding.image import ImageEmbedding


def test_image_shape_norm():
    _url = 'file:' + pathname2url(os.path.join(os.path.dirname(__file__), '../rose.jpg'))

    e = ImageEmbedding()

    v = e.transform(url=_url)
    assert tuple(v.shape) == (e.get_dim(),)
    assert abs(float(np.linalg.norm(v, ord=2, axis=None, keepdims=False) - 1.)) < 1.0e-5


def test_image_batching():
    _url = 'file:' + pathname2url(os.path.join(os.path.dirname(__file__), '../rose.jpg'))

    e = ImageEmbedding()

    _url_data_pairs = [(_url, None) for _ in range(3)]
    vv = e.transform_batch(_url_data_pairs)
    for v in vv:
        assert tuple(v.shape) == (e.get_dim(),)
        assert abs(float(np.linalg.norm(v, ord=2, axis=None, keepdims=False) - 1.)) < 1.0e-5
