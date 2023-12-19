from datetime import datetime
from os import environ
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from financegpt.data.data_connector import DATA_COLLECTION
from financegpt.data.data_connector import MongoDBConnector
from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.data.dataset import Dataset


@pytest.fixture
def test_data_date_start() -> datetime:
    return datetime(2021, 1, 1)


@pytest.fixture
def test_data_date_end() -> datetime:
    return datetime(2021, 1, 10)


@pytest.fixture
def test_text_data_df(
    test_data_date_start: datetime, test_data_date_end: datetime
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "symbol": ["djia"] * 10,
            "text": [rng.choice(["test", "test2", "test3"]) for _ in range(10)],
            "timestamp": pd.date_range(
                start=test_data_date_start, end=test_data_date_end, freq="D"
            ),
        }
    )


@pytest.fixture
def test_ohlc_data_df(
    test_data_date_start: datetime, test_data_date_end: datetime
) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    open_prices = rng.uniform(0, 5, 10)
    high_prices = open_prices + rng.uniform(0, 1, 10)
    low_prices = open_prices - rng.uniform(0, 1, 10)
    close_prices = open_prices

    return pd.DataFrame(
        {
            "symbol": ["djia"] * 10,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": rng.integers(0, 100, 10),
            "timestamp": pd.date_range(
                start=test_data_date_start, end=test_data_date_end, freq="D"
            ),
        }
    )


@pytest.fixture
def connection_kwargs() -> dict[str, str | int]:
    return {
        "username": environ["FINGPT_DB_USERNAME"],
        "password": environ["FINGPT_DB_PASSWORD"],
        "host": environ["FINGPT_DB_HOST"],
        "port": int(environ["FINGPT_DB_PORT"]),
        "db_name": environ["FINGPT_DB_NAME"],
    }


@pytest.fixture
def db_connector(connection_kwargs) -> Generator[MongoDBConnector, None, None]:
    with MongoDBConnector(**connection_kwargs) as connector:
        yield connector


@pytest.fixture
def data_populated_db(
    db_connector: MongoDBConnector, text_data_path: str, ohlc_data_path: str
):
    """
    Populates the database with data from the specified paths.
    """
    # Get the data from the files.
    text_data = pd.read_csv(text_data_path)
    ohlc_data = pd.read_csv(ohlc_data_path)

    # TODO: Move this functionality to Dataset class.
    text_dataset = Dataset(
        [TextDataPoint(**row.to_dict()) for _, row in text_data.iterrows()]
    )
    ohlc_dataset = Dataset(
        [OhlcDataPoint(**row.to_dict()) for _, row in ohlc_data.iterrows()]
    )

    try:
        db_connector.store_dataset(text_dataset)
        db_connector.store_dataset(ohlc_dataset)
        yield db_connector
    finally:
        db_connector._client[db_connector._db_name][DATA_COLLECTION].delete_many({})
