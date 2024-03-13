from typing import Generic
from typing import Iterator
from typing import overload
from typing import SupportsIndex
from typing import TypeVar

import pandas as pd

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

    def __iter__(self) -> Iterator[DataPointType]:
        return iter(self._data)

    def __add__(self, other: "Dataset[DataPointType]") -> "Dataset[DataPointType]":
        return Dataset(self._data + other.data)

    def __repr__(self) -> str:
        return repr(self._data)

    def to_dataframe(self):
        return pd.DataFrame([data_point.model_dump() for data_point in self._data])
