from hashlib import md5

import numpy as np

from .abc import Embedding


class NonLSHEmbedding(Embedding):
    
    def __init__(self, dim: int):
        self.__dim = dim

    def get_dim(self) -> int:
        return self.__dim

    def get_version(self) -> str:
        return f'aptl3/r-{self.__dim:d}'

    def transform(self, *, url: str = None, data: bytes = None) -> np.ndarray:
        if data is None:
            raise NotImplementedError
        h = md5()
        h.update(data)
        _hash = h.digest()
        seed = int.from_bytes(_hash, 'little', signed=False)
        rng = np.random.Generator(np.random.SFC64(seed=seed))
        v = rng.normal(0, 1, size=[self.__dim])
        v /= np.sum(v**2)**0.5  # L2 normalization
        return v

