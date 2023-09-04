from datetime import datetime

import pytest

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint


@pytest.fixture
def ohlc_data_point():
    return OhlcDataPoint(
        symbol="AAPL",
        timestamp=datetime(2021, 1, 1),
        interval="W",
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10000,
    )


@pytest.fixture
def text_data_point():
    return TextDataPoint(
        symbol="AAPL",
        timestamp=datetime(2021, 1, 1),
        interval="W",
        text="This is a test",
    )


@pytest.fixture
def ohlc_template():
    return (
        "What is the performance of {datapoint_symbol} on {datapoint_timestamp}"
        + " with interval {datapoint_interval}?\nThe performance of"
        + " {datapoint_symbol} on {datapoint_timestamp} ({datapoint_interval})"
        + " is {datapoint_open} {datapoint_high} {datapoint_low}"
        + " {datapoint_close} {datapoint_volume}"
    )


@pytest.fixture
def text_template():
    return (
        "What is the news for {datapoint_symbol} on {datapoint_timestamp}"
        + " with interval {datapoint_interval}?\nThe news for"
        + " {datapoint_symbol} on {datapoint_timestamp} ({datapoint_interval})"
        + " is {datapoint_text}"
    )
