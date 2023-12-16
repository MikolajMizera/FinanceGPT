import datetime

import pytest

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.data.dataset import Dataset
from financegpt.template.data_container import TemplateDataContainerFactory
from financegpt.template.templates import SimpleTemplateMeta


@pytest.fixture
def text_dataset_5days():
    return Dataset(
        data=[
            TextDataPoint(
                text="This is a test",
                timestamp=datetime.datetime(2021, 1, 1) + datetime.timedelta(days=i),
                symbol="AAPL",
                interval="W",
            )
            for i in range(5)
        ]
    )


@pytest.fixture
def container_factory_2d():
    return TemplateDataContainerFactory(window_size=2)


@pytest.mark.parametrize(
    "window,expected",
    [
        (1, 5),
        (2, 4),
        (3, 3),
        (4, 2),
        (5, 1),
    ],
)
def test_window_size(
    window: int, expected: int, ohlc_dataset_5days: Dataset[OhlcDataPoint]
):
    container_factory_2d = TemplateDataContainerFactory(window_size=window)
    containers = container_factory_2d._get_next_data_points(ohlc_dataset_5days)
    assert len(list(containers)) == expected


@pytest.mark.parametrize(
    "dataset_fixture,template_fixture",
    [
        ("ohlc_dataset_5days", "ohlc_template_meta"),
        ("text_dataset_5days", "text_template_meta"),
    ],
)
def test_prompt_factory(
    request,
    dataset_fixture: str,
    template_fixture: str,
    container_factory_2d: TemplateDataContainerFactory,
):
    dataset = request.getfixturevalue(dataset_fixture)
    template_metadata: SimpleTemplateMeta = request.getfixturevalue(template_fixture)

    prompts = container_factory_2d.create_containers(
        template=template_metadata,
        dataset=dataset,
    )
    assert (
        "2021-01-04 00:00:00" in prompts[-1].format_prompt()
        and "2021-01-05 00:00:00" in prompts[-1].format_prompt()
    )
