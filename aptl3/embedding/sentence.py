import logging
import warnings
from typing import Iterable, Optional, List, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .abc import Embedding, RequestIgnored


class SentenceEmbedding(Embedding):

    def __init__(self):
        logging.debug(f'torch.__config__.parallel_info():\n {torch.__config__.parallel_info()}')
        self.__model_name = 'distiluse-base-multilingual-cased'
        # self.__text_embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        self.__text_embed = SentenceTransformer(self.__model_name)
        self.__dim = self.__text_embed[-1].out_features

    def get_dim(self) -> int:
        return self.__dim

    def get_version(self) -> str:
        return 'sentence_transformers/' + self.__model_name

    @property
    def batch_size(self) -> int:
        return 32

    def transform(self, *, url: str = None, data: bytes = None) -> np.ndarray:
        x = self._transform_1(url=url)
        x = self._transform_2([x])
        return x[0]

    def _transform_1(self, *, url: str = None) -> str:
        if url is None:
            raise NotImplementedError
        if url.startswith('data:,'):
            text = url.split(',', maxsplit=1)[1]
        else:
            raise RequestIgnored('Not a sentence')
        if len(text) > self.__text_embed.get_max_seq_length():
            warnings.warn('Text truncated while embedding')
        return text

    def _transform_2(self, texts: List[str]):
        vv = self.__text_embed.encode(texts, batch_size=min(32, len(texts)), show_progress_bar=False)

        assert len(vv.shape) == 2 and vv.shape[0] == len(texts) and vv.shape[1] == self.__dim

        vv /= np.sum(vv ** 2, axis=1, keepdims=True) ** 0.5  # L2 normalization
        return vv

    def batching_map(self, *, url: Optional[str] = None, data: Optional[bytes] = None):
        return self._transform_1(url=url)

    def batching_transform(self, *, batch: Any) -> Iterable[np.ndarray]:
        return self._transform_2(batch)
