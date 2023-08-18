from datetime import datetime

import pytest

from financegpt.data.data_point import OhlcDataPoint, TextDataPoint
from financegpt.data.dataset import Dataset


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


def test_text_dataset_creation(text_data_point):
    text_dataset = Dataset(data=[text_data_point])
    assert len(text_dataset) == 1
    assert text_dataset[0] == text_data_point


def test_ohlc_dataset_creation(ohlc_data_point):
    ohlc_dataset = Dataset(data=[ohlc_data_point])
    assert len(ohlc_dataset) == 1
    assert ohlc_dataset[0] == ohlc_data_point
