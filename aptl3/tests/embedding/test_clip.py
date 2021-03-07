import os
import numpy as np
from urllib.request import pathname2url
import pytest

try:
    import clip
except ImportError:
    _NO_CLIP = True
else:
    _NO_CLIP = False
    from ...embedding.openai_clip import ClipTextEmbedding, ClipImageEmbedding


@pytest.mark.skipif(_NO_CLIP, reason='clip-by-openai not installed.')
def test_image_shape_norm():
    _url = 'file:' + pathname2url(os.path.join(os.path.dirname(__file__), '../rose.jpg'))

    e = ClipImageEmbedding()

    v = e.transform(url=_url)
    assert tuple(v.shape) == (e.get_dim(),)
    assert abs(float(np.linalg.norm(v, ord=2, axis=None, keepdims=False) - 1.)) < 1.0e-5


@pytest.mark.skipif(_NO_CLIP, reason='clip-by-openai not installed.')
def test_text_batching():
    _urls = [
        'data:,This is a sentence for the test case.',
        'data:,The present one is a test sentence.',
    ]

    e = ClipTextEmbedding()

    _url_data_pairs = [(_url, None) for _url in _urls * 2]
    vv = e.transform_batch(_url_data_pairs)
    for v in vv:
        assert tuple(v.shape) == (e.get_dim(),)
        assert abs(float(np.linalg.norm(v, ord=2, axis=None, keepdims=False) - 1.)) < 1.0e-5


if __name__ == '__main__':
    test_image_shape_norm()
    test_text_batching()
