import pytest

from financegpt.data.data_connector import MongoDBConnector
from financegpt.data.data_connector import TEMPLATES_COLLECTION


@pytest.mark.parametrize(
    "template_data_fixture",
    [
        "ohlc_template_meta",
        "text_template_meta",
        "ohlc_chat_template_meta",
        "text_chat_template_meta",
    ],
)
def test_store_prompt_templates(
    db_connector: MongoDBConnector, template_data_fixture, request
):
    template_data = request.getfixturevalue(template_data_fixture)
    try:
        db_connector.store_templates([template_data])
        recieved_templates = db_connector.get_templates()
        assert len(recieved_templates) == 1
    finally:
        db_connector._client[db_connector._db_name][TEMPLATES_COLLECTION].delete_many(
            {}
        )
