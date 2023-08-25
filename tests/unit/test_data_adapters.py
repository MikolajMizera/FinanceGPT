from datetime import datetime

import pandas as pd
import pytest

from financegpt.data.data_adapter import CSVOhlcDataAdapter
from financegpt.data.data_adapter import YahooOhlcApiDataAdapter
from financegpt.data.utils import format_date


def test_date_formatting():
    assert format_date(datetime(2021, 1, 1)) == "2021-01-01"
    formatted_date = format_date(datetime(2021, 1, 1, 12, 0, 0), interval="H1")
    assert formatted_date == "2021-01-01 12:00:00"


@pytest.fixture
def yahoo_adapter():
    return YahooOhlcApiDataAdapter()


@pytest.mark.parametrize("symbol", YahooOhlcApiDataAdapter.AVAIABLE_SYMBOLS)
def test_yahoo_adapter_avaiable_symbols(
    symbol: str, yahoo_adapter: YahooOhlcApiDataAdapter
):
    dataset = yahoo_adapter.get_data(
        symbol, datetime(2021, 8, 16), datetime(2021, 8, 17), "D"
    )
    assert len(dataset) == 2


def test_csv_adapter(tmp_path):
    pd.DataFrame(
        data={
            "Open": [1.0] * 2,
            "High": [1.0] * 2,
            "Low": [1.0] * 2,
            "Close": [1.0] * 2,
            "Adj Close": [1.0] * 2,
            "Volume": [1] * 2,
        },
        index=[datetime(2021, 8, 16), datetime(2021, 8, 17)],
    ).to_csv(tmp_path / "AAPL.csv")
    adapter = CSVOhlcDataAdapter(tmp_path, index_col=0)
    dataset = adapter.get_data(
        "AAPL", datetime(2021, 8, 16), datetime(2021, 8, 17), "D"
    )
    assert len(dataset) == 2
