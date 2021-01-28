from abc import ABC, abstractmethod
from typing import Iterable, Optional

from numpy import ndarray


class RequestIgnored(Exception):
    pass


class Embedding(ABC):

    @abstractmethod
    def get_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_version(self) -> str:
        """{python_module}/{handle}"""
        raise NotImplementedError

    @abstractmethod
    def transform(self, *, url: Optional[str] = None, data: Optional[bytes] = None) -> ndarray:
        raise NotImplementedError

    def transform_batch(self, urls: Iterable[str]) -> Iterable[ndarray]:
        for url in urls:
            yield self.transform(url=url)
