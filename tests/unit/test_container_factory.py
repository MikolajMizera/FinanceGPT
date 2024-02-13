import datetime

import pytest

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.data.dataset import Dataset
from financegpt.template.data_container import TemplateDataContainerFactory


@pytest.fixture
def text_dataset_5days_weekend():
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
    window: int,
    expected: int,
    ohlc_dataset_5days_weekend: Dataset[OhlcDataPoint],
    templates: dict,
):
    container_factory = TemplateDataContainerFactory(
        window_size=window,
        example_template=templates["example"],
        ohlc_template=templates["ohlc"],
        text_template=templates["text"],
    )

    date_index = [dp.timestamp for dp in ohlc_dataset_5days_weekend]
    containers = container_factory._get_next_window(date_index)
    assert len(list(containers)) == expected


def test_container_factory(
    ohlc_dataset_5days_weekend: Dataset[OhlcDataPoint],
    text_dataset_5days_weekend: Dataset[TextDataPoint],
    container_factory_2d: TemplateDataContainerFactory,
):
    containers = container_factory_2d.data_windows(
        ohlc_dataset=ohlc_dataset_5days_weekend, text_dataset=text_dataset_5days_weekend
    )
    assert (
        "2021-01-04 00:00:00" in containers[-1].format_prompt()
        and "2021-01-05 00:00:00" in containers[-1].format_prompt()
    )
