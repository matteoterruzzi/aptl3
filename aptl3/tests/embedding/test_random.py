import numpy as np

from ...embedding.random import NonLSHEmbedding


def test_random_dim_shape_norm():
    for _dim in [1, 42, 2048]:

        e = NonLSHEmbedding(_dim)

        assert e.get_dim() == _dim
        for _data in [b"", b" ", b"this is a test sentence"]:
            v = e.transform(data=_data)
            assert tuple(v.shape) == (_dim,)
            assert abs(float(np.linalg.norm(v, ord=2, axis=None, keepdims=False) - 1.)) < 1.0e-5


def test_batching():
    _dim = 20
    e = NonLSHEmbedding(_dim)
    _url_data_pairs = [(None, _data) for _data in [b"", b" ", b"this is a test sentence"]]
    vv = e.transform_batch(_url_data_pairs)
    for v in vv:
        assert tuple(v.shape) == (_dim,)
        assert abs(float(np.linalg.norm(v, ord=2, axis=None, keepdims=False) - 1.)) < 1.0e-5
