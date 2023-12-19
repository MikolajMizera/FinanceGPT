from datetime import datetime

import pytest

from financegpt.app import AppController
from financegpt.app import RequestModel
from financegpt.data.data_connector import MongoDBConnector


@pytest.fixture
def example_request() -> RequestModel:
    return RequestModel(
        historical_data_start_date=datetime(2021, 1, 1),
        historical_data_end_date=datetime(2021, 1, 10),
        historical_data_interval="D",
        prediction_symbol="djia",
        prediction_end_date=datetime(2021, 1, 11),
    )


@pytest.mark.skip(reason="Integration test not ready yet")
def test_app_controler_integration(
    data_populated_db: MongoDBConnector,
    connection_kwargs: dict[str, str | int],
    example_request: RequestModel,
):
    """
    Tests the app controller.
    """
    app = AppController(
        llm_model="gpt-3.5-turbo-instruct",
        connection_kwargs=connection_kwargs,
        window_size=2,
    )
    inference = app.process_request(example_request)
    assert len(inference) > 0
