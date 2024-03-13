from datetime import datetime
from typing import Any

from financegpt.app import RequestModel


def test_request_parsing(example_request: dict[str, Any]):
    parsed = RequestModel(**example_request)
    assert parsed.prediction_symbol == "AAPL"
    assert parsed.prediction_end_date == datetime(2023, 12, 31)
    assert parsed.historical_data_start_date == datetime(2021, 1, 1)
    assert parsed.historical_data_end_date == datetime(2021, 12, 17)
    assert parsed.historical_data_interval == "W"
