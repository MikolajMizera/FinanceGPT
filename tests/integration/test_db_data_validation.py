from datetime import datetime
from os import environ

import pytest
from pymongo import MongoClient, errors


@pytest.fixture
def db_connection():
    yield MongoClient(
        f"mongodb://{environ['FINGPT_DB_HOST']}:{environ['FINGPT_DB_PORT']}/"
    )[environ["FINGPT_DB_NAME"]]


def test_validate_text_data_point_insert(db_connection):
    db_connection.data_points.insert_one(
        {
            "type": "text",
            "symbol": "djia",
            "text": "test",
            "time": datetime(2021, 1, 1),
        }
    )


def test_validate_ohlc_data_points_insert(db_connection):
    db_connection.data_points.insert_one(
        {
            "type": "ohlc",
            "symbol": "djia",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1,
            "time": datetime(2021, 1, 1),
        }
    )


def test_validate_ohlc_data_points_insert_with_missing(db_connection):
    db_connection.data_points.insert_one(
        {
            "type": "ohlc",
            "symbol": "djia",
            "open": None,
            "high": None,
            "low": None,
            "close": 1.0,
            "volume": None,
            "time": datetime(2021, 1, 1),
        }
    )


def test_incorrect_data_point_type(db_connection):
    with pytest.raises(errors.WriteError):
        db_connection.data_points.insert_one(
            {
                "type": "incorrect",
                "symbol": "djia",
                "text": "test",
                "time": datetime(2021, 1, 1),
            }
        )


def test_incorrect_ohlc_data_points_insert(db_connection):
    with pytest.raises(errors.WriteError):
        db_connection.data_points.insert_one(
            {
                "type": "ohlc",
                "symbol": "djia",
                "open": "wrong_type",
                "high": 1.0,
                "low": 1.0,
                "time": datetime(2021, 1, 1),
            }
        )


def test_incomplete_ohlc_data_points_insert(db_connection):
    with pytest.raises(errors.WriteError):
        db_connection.data_points.insert_one(
            {
                "type": "ohlc",
                "symbol": "djia",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "time": datetime(2021, 1, 1),
            }
        )
