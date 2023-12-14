import pytest
from langchain.prompts.prompt import PromptTemplate

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.prompting.prompt import Prompt
from financegpt.prompting.prompt import TemplateData


@pytest.fixture
def expected_ohlc_prompt():
    return (
        "What is the performance of AAPL on 2021-01-01 00:00:00 with interval W?"
        + "\nThe performance of AAPL on 2021-01-01 00:00:00 (W) is 1.0 2.0 0.5"
        + " 1.5 10000"
    )


@pytest.fixture
def expected_text_prompt():
    return (
        "What is the news for AAPL on 2021-01-01 00:00:00 with interval W?"
        + "\nThe news for AAPL on 2021-01-01 00:00:00 (W) is This is a test"
    )


def test_ohlc_template_parsing(
    ohlc_data_point: OhlcDataPoint,
    ohlc_template_data: TemplateData,
    expected_ohlc_prompt: str,
):
    example_prompt = PromptTemplate(
        input_variables=ohlc_template_data.input_variables,
        template=ohlc_template_data.template,
    )

    prompt_str = example_prompt.format(
        **ohlc_data_point.dict_for_template(prefix="datapoint_")
    )
    assert prompt_str == expected_ohlc_prompt


def test_text_template_parsing(
    text_data_point: TextDataPoint,
    text_template_data: TemplateData,
    expected_text_prompt: str,
):
    example_prompt = PromptTemplate(
        input_variables=text_template_data.input_variables,
        template=text_template_data.template,
    )

    prompt_str = example_prompt.format(
        **text_data_point.dict_for_template(prefix="datapoint_")
    )
    assert prompt_str == expected_text_prompt


def test_create_prompt_ohlc(
    ohlc_data_point: OhlcDataPoint,
    ohlc_template_data: TemplateData,
    expected_ohlc_prompt: str,
):
    data_point_dict = ohlc_data_point.dict_for_template()
    prompt = Prompt(
        template=PromptTemplate(
            input_variables=ohlc_template_data.input_variables,
            template=ohlc_template_data.template,
        ),
        template_data=[data_point_dict],
    )
    prompt_str = prompt.format_prompt()
    assert prompt_str == expected_ohlc_prompt


def test_create_prompt_text(
    text_data_point: TextDataPoint,
    text_template_data: TemplateData,
    expected_text_prompt: str,
):
    data_point_dict = text_data_point.dict_for_template()
    prompt = Prompt(
        template=PromptTemplate(
            input_variables=text_template_data.input_variables,
            template=text_template_data.template,
        ),
        template_data=[data_point_dict],
    )
    prompt_str = prompt.format_prompt()
    assert prompt_str == expected_text_prompt


def test_create_prompt_ohlc_multiple(
    ohlc_data_point: OhlcDataPoint,
    ohlc_template_data: TemplateData,
    expected_ohlc_prompt: str,
):
    data_point_dict = ohlc_data_point.dict_for_template()
    prompt = Prompt(
        template=PromptTemplate(
            input_variables=ohlc_template_data.input_variables,
            template=ohlc_template_data.template,
        ),
        template_data=[data_point_dict, data_point_dict],
    )
    prompt_str = prompt.format_prompt()
    assert prompt_str == (expected_ohlc_prompt + "\n" + expected_ohlc_prompt)


def test_create_prompt_text_multiple(
    text_data_point: TextDataPoint,
    text_template_data: TemplateData,
    expected_text_prompt: str,
):
    data_point_dict = text_data_point.dict_for_template()
    prompt = Prompt(
        template=PromptTemplate(
            input_variables=text_template_data.input_variables,
            template=text_template_data.template,
        ),
        template_data=[data_point_dict, data_point_dict],
    )
    prompt_str = prompt.format_prompt()
    assert prompt_str == (expected_text_prompt + "\n" + expected_text_prompt)


def test_create_prompt_invalid_data_type(
    ohlc_data_point: OhlcDataPoint,
    text_data_point: TextDataPoint,
    text_template_data: TemplateData,
):
    with pytest.raises(AssertionError):
        prompt = Prompt(
            template=PromptTemplate(
                input_variables=text_template_data.input_variables,
                template=text_template_data.template,
            ),
            template_data=[ohlc_data_point.dict_for_template()],
        )
        _ = prompt.format_prompt()
