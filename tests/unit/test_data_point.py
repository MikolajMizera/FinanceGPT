from datetime import datetime

import pytest
from pydantic import ValidationError

from financegpt.data.data_point import OhlcDataPoint, TextDataPoint


def test_text_data_point_creation():
    text_data_point = TextDataPoint(
        symbol="AAPL",
        timestamp=datetime(2021, 1, 1),
        interval="W",
        text="This is a test",
    )
    assert text_data_point.symbol == "AAPL"
    assert text_data_point.timestamp == datetime(2021, 1, 1)
    assert text_data_point.interval == "W"
    assert text_data_point.text == "This is a test"


def test_ohlc_data_point_creation():
    ohlc_data_point = OhlcDataPoint(
        symbol="AAPL",
        timestamp=datetime(2021, 1, 1),
        interval="W",
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10000,
    )
    assert ohlc_data_point.symbol == "AAPL"
    assert ohlc_data_point.timestamp == datetime(2021, 1, 1)
    assert ohlc_data_point.interval == "W"
    assert ohlc_data_point.open == 1.0
    assert ohlc_data_point.high == 2.0
    assert ohlc_data_point.low == 0.5
    assert ohlc_data_point.close == 1.5
    assert ohlc_data_point.volume == 10000


def test_invalid_type_rise_error():
    with pytest.raises(ValidationError):
        _ = OhlcDataPoint(
            symbol="AAPL",
            timestamp=datetime(2021, 1, 1),
            interval="W",
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            volume=10.8,
        )  # type: ignore
