from typing import Generic
from typing import overload
from typing import SupportsIndex
from typing import TypeVar

from .data_point import DataPoint

__all__ = ["Dataset"]

DataPointType = TypeVar("DataPointType", bound=DataPoint, covariant=True)


class Dataset(Generic[DataPointType]):
    def __init__(self, data: list[DataPointType]):
        self._data = data

    @property
    def data(self) -> list[DataPointType]:
        return self._data

    @overload
    def __getitem__(self, index: SupportsIndex) -> DataPointType:
        ...

    @overload
    def __getitem__(self, index: slice) -> "Dataset[DataPointType]":
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Dataset(data=self._data[index])
        else:
            return self._data[index]

    def __len__(self) -> int:
        return len(self._data)
