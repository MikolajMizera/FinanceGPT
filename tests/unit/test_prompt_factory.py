import datetime

import pytest

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.data.dataset import Dataset
from financegpt.prompting.prompt import PromptFactory
from financegpt.prompting.prompt import PromptTemplate
from financegpt.prompting.prompt import RegularTemplateData


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
    ohlc_dataset_5days: Dataset[OhlcDataPoint],
):
    prompt_factory_2d = PromptFactory(window_size=window)
    prompts = prompt_factory_2d._get_next_data_points(ohlc_dataset_5days)
    assert len(list(prompts)) == expected


@pytest.mark.parametrize(
    "dataset_fixture,template_fixture",
    [
        ("ohlc_dataset_5days", "ohlc_template_data"),
        ("text_dataset_5days", "text_template_data"),
    ],
)
def test_prompt_factory(request, dataset_fixture: str, template_fixture: str):
    dataset = request.getfixturevalue(dataset_fixture)
    template_data: RegularTemplateData = request.getfixturevalue(template_fixture)

    prompt_factory_2d = PromptFactory(window_size=2)
    prompts = prompt_factory_2d.create_prompts(
        template=PromptTemplate(
            input_variables=template_data.input_variables,
            template=template_data.template,
        ),
        dataset=dataset,
    )
    assert (
        "2021-01-04 00:00:00" in prompts[-1].format_prompt()
        and "2021-01-05 00:00:00" in prompts[-1].format_prompt()
    )
