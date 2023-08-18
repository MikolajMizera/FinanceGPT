from abc import ABC
from typing import Generic, List, TypeVar

from .data_point import DataPoint

__all__ = ["Dataset"]

DataPointType = TypeVar("DataPointType", bound=DataPoint)


class Dataset(ABC, Generic[DataPointType]):
    def __init__(self, data: List[DataPointType]):
        self._data = data

    @property
    def data(self) -> List[DataPointType]:
        return self._data

    def __getitem__(self, index: int) -> DataPointType:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)
