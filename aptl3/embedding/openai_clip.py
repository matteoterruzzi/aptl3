"""
Multimodal embedding model using `"Contrastive Language-Image Pre-Training" by OpenAI
<https://github.com/openai/CLIP>`_
"""
import threading
import warnings
from typing import Iterable, Optional, List, Any, NamedTuple

import numpy as np
import torch

from .abc import Embedding, RequestIgnored
from ..images import getImage, ImageError


_MODEL_TUPLE = None
_MODEL_TUPLE_LOCK = threading.Lock()


class ClipModelTuple(NamedTuple):
    name: str
    device: str
    model: Any
    preprocess: Any
    tokenize: Any
    dim: int

    @staticmethod
    def load(name: Optional[str] = None) -> 'ClipModelTuple':
        """Load the named CLIP model by OpenAI (defaults to ViT-B/32)"""
        # noinspection PyPackageRequirements
        import clip

        name = name or "ViT-B/32"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(name, device=device)

        return ClipModelTuple(name, device, model, preprocess, clip.tokenize, 512)


class _ClipModelMixin:

    def __init__(self, model_tuple: Optional[ClipModelTuple] = None):
        if model_tuple is None:
            global _MODEL_TUPLE, _MODEL_TUPLE_LOCK
            with _MODEL_TUPLE_LOCK:
                # The lock avoids concurrent threads to repeatedly load the model
                if _MODEL_TUPLE is None:
                    _MODEL_TUPLE = ClipModelTuple.load()
            model_tuple = _MODEL_TUPLE

        (self.__model_name,
         self._device,
         self._model,
         self._preprocess,
         self._tokenize,
         self._dim
         ) = model_tuple

    def get_dim(self) -> int:
        return self._dim

    def get_version(self) -> str:
        return 'github.com/openai/CLIP/' + self.__model_name

    def get_space(self) -> Optional[str]:
        return 'github.com/openai/CLIP/' + self.__model_name


class ClipTextEmbedding(_ClipModelMixin, Embedding):

    def get_modality(self) -> Optional[str]:
        return 'text'

    @property
    def batch_size(self) -> int:
        return 32

    def transform(self, *, url: str = None, data: bytes = None) -> np.ndarray:
        x = self.batching_map(url=url, data=data)
        vv = self.batching_transform(batch=[x])
        return next(iter(vv))

    def batching_map(self, *, url: Optional[str] = None, data: Optional[bytes] = None):
        if url is None:
            raise NotImplementedError
        if not url.startswith('data:,'):
            raise RequestIgnored('Not a text')

        text = url.split(',', maxsplit=1)[1]
        try:
            text = self._tokenize([text])  # default context_length=77
        except RuntimeError as ex:
            # RuntimeError is raised if the text is too long for context length
            _err = str(ex)
            _too_long_err = 'is too long for context length'
            if _too_long_err in _err:
                warnings.warn('Ignoring text too long for CLIP model')
                raise RequestIgnored(f'Text {_err[_err.rfind(_too_long_err):]}') from ex
            else:
                raise RequestIgnored(ex) from ex
        assert tuple(text.shape) == (1, 77), text.shape
        return text

    def batching_transform(self, *, batch: List[Any]) -> Iterable[np.ndarray]:
        batch = torch.cat(batch, 0).to(self._device)
        assert len(tuple(batch.shape)) == 2, batch.shape

        with torch.no_grad():
            vv = self._model.encode_text(batch).detach().numpy()

        assert tuple(vv.shape) == (len(batch), self._dim), vv.shape

        vv /= np.sum(vv ** 2, axis=1, keepdims=True) ** 0.5  # L2 normalization
        return vv


class ClipImageEmbedding(_ClipModelMixin, Embedding):

    def get_modality(self) -> Optional[str]:
        return 'image'

    @property
    def batch_size(self) -> int:
        return 4

    def transform(self, *, url: str = None, data: bytes = None) -> np.ndarray:
        x = self.batching_map(url=url, data=data)
        vv = self.batching_transform(batch=[x])
        return next(iter(vv))

    def batching_map(self, *, url: Optional[str] = None, data: Optional[bytes] = None):
        assert url is not None or data is not None, (url, data)
        fp = data if data is not None else url
        try:
            image = getImage(fp)
        except ImageError as ex:
            raise RequestIgnored(ex) from ex
        image = self._preprocess(image.convert('RGB')).unsqueeze(0)
        assert tuple(image.shape) == (1, 3, 224, 224), image.shape
        return image

    def batching_transform(self, *, batch: List[Any]) -> Iterable[np.ndarray]:
        batch = torch.cat(batch, 0).to(self._device)
        assert len(tuple(batch.shape)) == 4, batch.shape

        with torch.no_grad():
            vv = self._model.encode_image(batch).detach().numpy()

        assert tuple(vv.shape) == (len(batch), self._dim), vv.shape

        vv /= np.sum(vv ** 2, axis=1, keepdims=True) ** 0.5  # L2 normalization
        return vv
