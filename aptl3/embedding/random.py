from typing import Optional
import numpy as np

from .abc import Embedding


class NonLSHEmbedding(Embedding):
    
    def __init__(self, dim: int):
        self.__dim = dim

    def get_dim(self) -> int:
        return self.__dim

    def transform(self, *, url: str = None, data: bytes = None) -> np.ndarray:
        seed = int.from_bytes(bytes(url, 'utf-8'), 'little', signed=False)
        rng = np.random.Generator(np.random.SFC64(seed=seed))
        v = rng.normal(0, 1, size=[self.__dim])
        v /= np.sum(v**2)**0.5  # L2 normalization
        return v

