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
