from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import ValidationError

__all__ = ["DataPoint", "OhlcDataPoint", "TextDataPoint", "IntervalType"]

IntervalType = Literal["W", "D", "H1"]


class DataPoint(BaseModel):
    symbol: str
    timestamp: datetime
    interval: IntervalType

    def __str__(self) -> str:
        raise NotImplementedError

    def dict_for_template(self, prefix="datapoint_") -> dict[str, str]:
        return {f"{prefix}{k}": str(v) for k, v in self.model_dump().items()}


class OhlcDataPoint(DataPoint):
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __str__(self) -> str:
        return f"""{self.symbol}\t{self.timestamp}\t{self.open}\t{self.high}\t
                {self.low}\t{self.close}\t{self.volume}\t{self.interval}"""


class TextDataPoint(DataPoint):
    text: str

    def __str__(self) -> str:
        return f"{self.symbol}\t{self.timestamp}\t{self.text}\t{self.interval}"


class DataPointFactory:
    @staticmethod
    def create_data_point(**kwargs) -> DataPoint:
        try:
            return OhlcDataPoint(**kwargs)
        except ValidationError:
            return TextDataPoint(**kwargs)
