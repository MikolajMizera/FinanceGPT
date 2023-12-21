from datetime import datetime
from datetime import timedelta

from financegpt.data.data_connector import MongoDBConnector
from financegpt.data.data_point import IntervalType


def get_data_from_db(
    data_populated_db: MongoDBConnector,
    symbol: str,
    test_data_date_start: datetime,
    test_data_date_end: datetime,
    interval: IntervalType,
):
    """
    Returns requested data.
    """
    start_date = test_data_date_start + timedelta(days=1)
    end_date = test_data_date_end - timedelta(days=1)
    dataset = data_populated_db.get_dataset(symbol, start_date, end_date, interval)

    assert dataset[0].timestamp == start_date
    assert dataset[-1].timestamp == end_date
    assert len(dataset) > 0
    assert all(dp.symbol == symbol for dp in dataset)
    assert all(dp.interval == interval for dp in dataset)
