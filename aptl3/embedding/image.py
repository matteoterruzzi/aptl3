import tensorflow_hub as hub
import numpy as np

from .abc import Embedding, RequestIgnored
from ..images import getSmallImage, imageToArray, ImageError


class ImageEmbedding(Embedding):

    def __init__(self):
        self.__model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1")
        self.__dim = 1280

    def get_dim(self) -> int:
        return self.__dim

    def transform(self, *, url: str = None, data: bytes = None) -> np.ndarray:

        assert url is not None or data is not None
        fp = data if data is not None else url

        if url is not None and url.startswith('data:,'):
            raise RequestIgnored('Not an image')

        try:
            img = imageToArray(getSmallImage(fp, 224, 224)).astype(np.float32) / 255.
        except ImageError as ex:
            raise RequestIgnored(ex) from ex
        
        v = self.__model([img])[0]
        
        assert len(v.shape) == 1 and v.shape[0] == self.__dim
        
        v /= np.sum(v ** 2) ** 0.5  # L2 normalization
        return v

    # TODO: reimplement transform_batch
