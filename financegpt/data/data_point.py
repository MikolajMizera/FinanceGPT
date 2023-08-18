from datetime import datetime
from typing import Literal

from pydantic import BaseModel

__all__ = ["DataPoint", "OhlcDataPoint", "TextDataPoint"]


class DataPoint(BaseModel):
    symbol: str
    timestamp: datetime
    interval: Literal["W", "D", "H1"]

    def __str__(self) -> str:
        raise NotImplementedError


class OhlcDataPoint(DataPoint):
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __str__(self) -> str:
        return f"{self.symbol}\t{self.timestamp}\t{self.open}\t{self.high}\t"

    "{self.low}\t{self.close}\t{self.volume}\t{self.interval}"


class TextDataPoint(DataPoint):
    text: str

    def __str__(self) -> str:
        return f"{self.symbol}\t{self.timestamp}\t{self.text}\t{self.interval}"
