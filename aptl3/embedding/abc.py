from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, Any

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

    @property
    def batch_size(self) -> int:
        return 1

    def transform_batch(self, urls_data_pairs: Iterable[Tuple[Optional[str], Optional[bytes]]]) -> Iterable[ndarray]:
        batch = None
        for url, data in urls_data_pairs:
            mapped = self.batching_map(url=url, data=data)
            batch = self.batching_reduce(batch=batch, mapped=mapped)
        yield from self.batching_transform(batch=batch)

    # noinspection PyMethodMayBeStatic
    def batching_map(self, *, url: Optional[str] = None, data: Optional[bytes] = None):
        return url, data

    # noinspection PyMethodMayBeStatic
    def batching_reduce(self, *, batch: Optional[Any] = None, mapped: Any) -> Any:
        if batch is None:
            return [mapped]
        else:
            return batch + [mapped]

    def batching_transform(self, *, batch: Any) -> Iterable[ndarray]:
        for url, data in batch:
            yield self.transform(url=url, data=data)
