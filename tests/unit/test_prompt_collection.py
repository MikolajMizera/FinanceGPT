import pytest

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.dataset import Dataset
from financegpt.prompting.prompt import Prompt
from financegpt.prompting.prompt import PromptCollection
from financegpt.prompting.prompt import PromptFactory
from financegpt.prompting.prompt import PromptTemplate
from financegpt.prompting.prompt import RegularTemplateData


@pytest.fixture
def prompts(
    ohlc_dataset_5days: Dataset[OhlcDataPoint], ohlc_template_data: RegularTemplateData
):
    prompt_factory_2d = PromptFactory(window_size=2)
    return prompt_factory_2d.create_prompts(
        template=PromptTemplate(
            input_variables=ohlc_template_data.input_variables,
            template=ohlc_template_data.template,
        ),
        dataset=ohlc_dataset_5days,
    )


def test_prompt_collection(prompts: list[Prompt]):
    prompt_collection = PromptCollection(prompts=prompts)
    prompt_str = prompt_collection.format_prompt()
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


def test_prompt_collection_add(prompts: list[Prompt]):
    prompt_collection = PromptCollection(prompts=prompts)
    new_collection = prompt_collection + prompt_collection
    # 4 windows * 2 prompts * 2 lines * 2 collections = 32 lines total
    assert len(new_collection.format_prompt().split("\n")) == 32
