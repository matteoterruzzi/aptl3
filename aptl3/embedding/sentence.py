import warnings
from typing import Iterable, Tuple

import numpy as np

from .abc import Embedding, RequestIgnored


class SentenceEmbedding(Embedding):

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.__model_name = 'distiluse-base-multilingual-cased'
        # self.__text_embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        self.__text_embed = SentenceTransformer(self.__model_name)
        self.__dim = self.__text_embed[-1].out_features

    def get_dim(self) -> int:
        return self.__dim

    def get_version(self) -> str:
        return 'sentence_transformers/' + self.__model_name

    def transform(self, *, url: str = None, data: bytes = None) -> np.ndarray:
        return next(iter(self.transform_batch([url])))

    def transform_batch(self, urls: Iterable[str]) -> Iterable[np.ndarray]:
        texts = []
        for url in urls:
            if url.startswith('data:,'):
                text = url.split(',', maxsplit=1)[1]
            else:
                # NOTE: it is counterintuitive to use the url as a sentence and it would be a problem for procrustean:
                #  it ends up aligning the image embedding with the embedding of the url instead of the text annotation!
                # TODO: make transform_batch generate RequestIgnored for individual elements (allow a retry for others)
                # NOTE: the point here is that a better embedding exists for urls pointing to images.
                #  Also, we hope that eventually we will not need to use the url text itself to make an embedding.
                #  A file path may still contain useful information, but it must be considered in a different way.
                raise RequestIgnored('Not a sentence')
            if len(text) > self.__text_embed.get_max_seq_length():
                warnings.warn('Text truncated while embedding')

            texts.append(text)

        vv = self.__text_embed.encode(texts, batch_size=min(32, len(texts)), show_progress_bar=False)

        assert len(vv.shape) == 2 and vv.shape[0] == len(texts) and vv.shape[1] == self.__dim

        vv /= np.sum(vv ** 2, axis=1, keepdims=True) ** 0.5  # L2 normalization
        return vv
