import logging
from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Generic
from typing import TypeVar

import pandas as pd
import yfinance as yf
from pandas_datareader import data as web

from .data_point import DataPoint
from .data_point import IntervalType
from .data_point import OhlcDataPoint
from .data_point import TextDataPoint
from .dataset import Dataset
from .utils import add_interval
from .utils import format_date

yf.pdr_override()

DataPointType = TypeVar("DataPointType", bound=DataPoint)
OHLC_REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PANDAS_INTERVALS: dict[IntervalType, str] = {
    "D": "D",
    "W": "W",
    "H1": "h",
}


class DataAdapter(ABC, Generic[DataPointType]):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def get_dataset(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: IntervalType,
    ) -> Dataset[DataPointType]:
        raise NotImplementedError


class YahooOhlcApiDataAdapter(DataAdapter[OhlcDataPoint]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    AVAIABLE_SYMBOLS = ("AAPL", "MSFT", "AMZN")

    def get_dataset(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: IntervalType,
    ) -> Dataset[OhlcDataPoint]:
        # add one period of interval to end_date to get the last data point
        end_date = add_interval(end_date, interval)
        data = web.get_data_yahoo(
            [symbol],
            start=format_date(start_date),
            end=format_date(end_date),
            **self.kwargs,
        )

        return Dataset(
            [
                OhlcDataPoint(
                    symbol=symbol,
                    timestamp=tmstmp,
                    interval=interval or "D",
                    open=open,
                    high=high,
                    low=low,
                    close=adj_close,
                    volume=volume,
                )
                for tmstmp, open, high, low, _, adj_close, volume in data.itertuples()
            ]
        )


class CSVOhlcDataAdapter(DataAdapter[OhlcDataPoint]):
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(**kwargs)
        self._data_dir = data_dir

    def get_dataset(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: IntervalType | None,
    ) -> Dataset[OhlcDataPoint]:
        path = f"{self._data_dir}/{symbol}.csv"
        data: pd.DataFrame = pd.read_csv(path, parse_dates=True, **self.kwargs)
        dates_mask = data.index.to_series().between(start_date, end_date)
        data = data.loc[dates_mask]

        logging.info(f"Loaded {len(data)} data points for {symbol}...")
        logging.debug(f"Data columns: {data.columns.to_list()}")
        assert all(c in data.columns for c in OHLC_REQUIRED_COLUMNS)

        return Dataset(
            [
                OhlcDataPoint(
                    symbol=symbol,
                    timestamp=tmstmp,
                    interval=interval or "D",
                    open=open,
                    high=high,
                    low=low,
                    close=adj_close,
                    volume=volume,
                )
                for tmstmp, open, high, low, _, adj_close, volume in data.itertuples()
            ]
        )


class CSVTextDataAdapter(DataAdapter[TextDataPoint]):
    def __init__(self, data_dir: str, merge_by_interval: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._data_dir = data_dir
        self._merge_by_interval = merge_by_interval

    def _group_by_interval(
        self, data: pd.DataFrame, interval: IntervalType
    ) -> pd.DataFrame:
        grouper = pd.Grouper(freq=PANDAS_INTERVALS.get(interval, "D"))
        grouped = (
            data.groupby(grouper).apply(lambda x: "#".join(x["Text"])).to_frame("Text")
        )
        return grouped[grouped["Text"].str.len() > 0]

    def get_dataset(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: IntervalType | None,
    ) -> Dataset[TextDataPoint]:
        path = f"{self._data_dir}/{symbol}.csv"
        data: pd.DataFrame = pd.read_csv(path, parse_dates=True, **self.kwargs)
        dates_mask = data.index.to_series().between(start_date, end_date)
        data = data.loc[dates_mask]

        if interval and self._merge_by_interval:
            data = (
                self._group_by_interval(data, interval)
                if self._merge_by_interval
                else data
            )

        assert data.columns.to_list() == ["Text"]

        return Dataset(
            [
                TextDataPoint(
                    symbol=symbol,
                    timestamp=timestamp,
                    interval=interval or "D",
                    text=text,
                )
                for timestamp, text in data.itertuples()
            ]
        )
