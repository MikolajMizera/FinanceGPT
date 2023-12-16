import pytest

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.dataset import Dataset
from financegpt.template.data_container import TemplateDataContainer
from financegpt.template.data_container import TemplateDataContainerCollection
from financegpt.template.data_container import TemplateDataContainerFactory
from financegpt.template.templates import SimpleTemplateMeta


@pytest.fixture
def template_data_containers(
    ohlc_dataset_5days: Dataset[OhlcDataPoint], ohlc_template_meta: SimpleTemplateMeta
):
    prompt_factory_2d = TemplateDataContainerFactory(window_size=2)
    return prompt_factory_2d.create_containers(
        template=ohlc_template_meta,
        dataset=ohlc_dataset_5days,
    )


def test_container_collection(template_data_containers: list[TemplateDataContainer]):
    containers_collection = TemplateDataContainerCollection(
        containers=template_data_containers
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
    template_data_containers: list[TemplateDataContainer],
):
    containers_collection = TemplateDataContainerCollection(
        containers=template_data_containers
    )
    new_collection = containers_collection + containers_collection
    # 4 windows * 2 prompts * 2 lines * 2 collections = 32 lines total
    assert len(new_collection.format_prompt().split("\n")) == 32
