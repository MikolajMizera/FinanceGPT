import pytest

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.dataset import Dataset
from financegpt.template.data_container import TemplateDataContainer
from financegpt.template.data_container import TemplateDataContainerCollection
from financegpt.template.data_container import TemplateDataContainerFactory
from financegpt.template.templates import ChatTemplateMeta
from financegpt.template.templates import SimpleTemplateMeta


@pytest.fixture
def template_data_containers_weekend(
    ohlc_dataset_5days_weekend: Dataset[OhlcDataPoint],
    templates: dict[str, SimpleTemplateMeta | ChatTemplateMeta],
):
    prompt_factory_2d = TemplateDataContainerFactory(
        window_size=2,
        example_template=templates["example"],
        ohlc_template=templates["ohlc"],
        text_template=templates["text"],
    )
    return prompt_factory_2d.data_windows(
        ohlc_dataset=ohlc_dataset_5days_weekend, text_dataset=None
    )


@pytest.fixture
def template_data_containers_weekday(
    ohlc_dataset_5days_weekday: Dataset[OhlcDataPoint],
    templates: dict[str, SimpleTemplateMeta | ChatTemplateMeta],
):
    prompt_factory_2d = TemplateDataContainerFactory(
        window_size=2,
        example_template=templates["example"],
        ohlc_template=templates["ohlc"],
        text_template=templates["text"],
    )
    return prompt_factory_2d.data_windows(
        ohlc_dataset=ohlc_dataset_5days_weekday, text_dataset=None
    )


def test_container_collection(
    template_data_containers_weekend: list[TemplateDataContainer],
):
    containers_collection = TemplateDataContainerCollection(
        containers=template_data_containers_weekend
    )
    prompt_str = containers_collection.format_prompt()
    assert all(
        date in prompt_str
        for date in [
            "2021-01-01",
            "2021-01-02",
            "2021-01-03",
            "2021-01-04",
            "2021-01-05",
        ]
    )


def test_container_collection_add(
    template_data_containers_weekday: list[TemplateDataContainer],
):
    containers_collection = TemplateDataContainerCollection(
        containers=template_data_containers_weekday
    )
    new_collection = containers_collection + containers_collection
    assert len(new_collection.format_prompt().split("\n")) == 48
