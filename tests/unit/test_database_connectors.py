from datetime import datetime
from unittest.mock import MagicMock

import pytest

from financegpt.data.data_connector import DATA_COLLECTION
from financegpt.data.data_connector import MongoDBConnector
from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.data.dataset import Dataset


@pytest.fixture
def ohlc_records():
    return [
        {
            "symbol": "AAPL",
            "timestamp": datetime(2021, 1, 1),
            "interval": "W",
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 10000,
        }
    ] * 5


@pytest.fixture
def text_records():
    return [
        {
            "symbol": "AAPL",
            "timestamp": datetime(2021, 1, 1),
            "interval": "W",
            "text": "This is a test",
        }
    ] * 5


@pytest.fixture
def ohlc_dataset(ohlc_records):
    return Dataset([OhlcDataPoint(**r) for r in ohlc_records])


@pytest.fixture
def text_dataset(text_records):
    return Dataset([TextDataPoint(**r) for r in text_records])


@pytest.fixture
def mocked_mongo_data_connector():
    mongo_data_connector = MongoDBConnector(
        username="username",
        password="password",
        host="host",
        port=1234,
        db_name="db_name",
    )
    mongo_data_connector._client = MagicMock()
    return mongo_data_connector


@pytest.mark.parametrize("dataset_name", ["ohlc_dataset", "text_dataset"])
def test_store_data_points(
    mocked_mongo_data_connector: MongoDBConnector, dataset_name: str, request
):
    dataset: Dataset = request.getfixturevalue(dataset_name)
    mocked_mongo_data_connector.store_dataset(dataset)
    assert len(
        mocked_mongo_data_connector._client["db_name"][
            DATA_COLLECTION
        ].insert_one.mock_calls
    ) == len(dataset)


@pytest.mark.parametrize(
    "records, dataset_name",
    [("ohlc_records", "ohlc_dataset"), ("text_records", "text_dataset")],
)
def test_get_data_points(
    mocked_mongo_data_connector: MongoDBConnector,
    records: str,
    dataset_name: str,
    request,
):
    records = request.getfixturevalue(records)
    dataset = request.getfixturevalue(dataset_name)

    mocked_mongo_data_connector._client["db_name"][
        DATA_COLLECTION
    ].find.return_value = records
    assert len(
        mocked_mongo_data_connector.get_dataset(
            "AAPL", datetime(2021, 1, 1), datetime(2021, 1, 5), "W"
        )
    ) == len(dataset)

    call_args = mocked_mongo_data_connector._client["db_name"][
        DATA_COLLECTION
    ].find.call_args[0][0]
    assert call_args["symbol"] == "AAPL"
    assert call_args["timestamp"]["$gte"] == datetime(2021, 1, 1)
    assert call_args["timestamp"]["$lte"] == datetime(2021, 1, 5)
    assert call_args["interval"] == "W"
