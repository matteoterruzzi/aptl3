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
