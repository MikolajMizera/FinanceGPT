from os import environ

import pytest

from financegpt.data.data_connector import MongoDBConnector
from financegpt.data.data_connector import TEMPLATES_COLLECTION


@pytest.mark.parametrize(
    "template_data_fixture", ["ohlc_template_data", "text_template_data"]
)
def test_store_prompt_templates(template_data_fixture, request):
    template_data = request.getfixturevalue(template_data_fixture)
    with MongoDBConnector(
        username=environ["FINGPT_DB_USERNAME"],
        password=environ["FINGPT_DB_PASSWORD"],
        host=environ["FINGPT_DB_HOST"],
        port=int(environ["FINGPT_DB_PORT"]),
        db_name=environ["FINGPT_DB_NAME"],
    ) as connector:
        try:
            connector.store_templates([template_data])
            recieved_templates = connector.get_templates()
            assert len(recieved_templates) == 1
        finally:
            connector._client[connector._db_name][TEMPLATES_COLLECTION].delete_many({})
