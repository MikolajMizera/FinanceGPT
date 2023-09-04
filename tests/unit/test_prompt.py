import pytest
from langchain.prompts.prompt import PromptTemplate

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.prompting.prompt import Prompt


@pytest.fixture
def ohlc_template():
    return (
        "What is the performance of {datapoint_symbol} on {datapoint_timestamp}"
        + " with interval {datapoint_interval}?\nThe performance of "
        + "{datapoint_symbol} on {datapoint_timestamp} ({datapoint_interval}) "
        + "is {datapoint_open} {datapoint_high} {datapoint_low} "
        + "{datapoint_close} {datapoint_volume}"
    )


@pytest.fixture
def text_template():
    return (
        "What is the news for {datapoint_symbol} on {datapoint_timestamp}"
        + " with interval {datapoint_interval}?\nThe news for "
        + "{datapoint_symbol} on {datapoint_timestamp} ({datapoint_interval}) "
        + "is {datapoint_text}"
    )


def test_ohlc_template_parsing(ohlc_data_point: OhlcDataPoint, ohlc_template: str):
    example_prompt = PromptTemplate(
        input_variables=[
            "datapoint_symbol",
            "datapoint_timestamp",
            "datapoint_interval",
            "datapoint_open",
            "datapoint_high",
            "datapoint_low",
            "datapoint_close",
            "datapoint_volume",
        ],
        template=ohlc_template,
    )

    prompt_str = example_prompt.format(
        **ohlc_data_point.dict_for_template(prefix="datapoint_")
    )
    assert (
        prompt_str
        == """What is the performance of AAPL on 2021-01-01 00:00:00 with interval W?
    The performance of AAPL on 2021-01-01 00:00:00 (W) is 1.0 2.0 0.5 1.5 10000"""
    )


def test_text_template_parsing(text_data_point: TextDataPoint, text_template: str):
    example_prompt = PromptTemplate(
        input_variables=[
            "datapoint_symbol",
            "datapoint_timestamp",
            "datapoint_interval",
            "datapoint_text",
        ],
        template=text_template,
    )

    prompt_str = example_prompt.format(
        **text_data_point.dict_for_template(prefix="datapoint_")
    )
    assert (
        prompt_str
        == """What is the news for AAPL on 2021-01-01 00:00:00 with interval W?
    The news for AAPL on 2021-01-01 00:00:00 (W) is This is a test"""
    )


def test_create_prompt_ohlc(ohlc_data_point: OhlcDataPoint, ohlc_template: str):
    prompt = Prompt(
        template=PromptTemplate(
            input_variables=list(ohlc_data_point.dict_for_template().keys()),
            template=ohlc_template,
        ),
        data_points=[ohlc_data_point],
    )
    prompt_str = prompt.format_prompt()
    assert (
        prompt_str
        == """What is the performance of AAPL on 2021-01-01 00:00:00 with interval W?
    The performance of AAPL on 2021-01-01 00:00:00 (W) is 1.0 2.0 0.5 1.5 10000"""
    )


def test_create_prompt_text(text_data_point: TextDataPoint, text_template: str):
    prompt = Prompt(
        template=PromptTemplate(
            input_variables=list(text_data_point.dict_for_template().keys()),
            template=text_template,
        ),
        data_points=[text_data_point],
    )
    prompt_str = prompt.format_prompt()
    assert (
        prompt_str
        == """What is the news for AAPL on 2021-01-01 00:00:00 with interval W?
    The news for AAPL on 2021-01-01 00:00:00 (W) is This is a test"""
    )


def test_create_prompt_ohlc_multiple(
    ohlc_data_point: OhlcDataPoint, ohlc_template: str
):
    prompt = Prompt(
        template=PromptTemplate(
            input_variables=list(ohlc_data_point.dict_for_template().keys()),
            template=ohlc_template,
        ),
        data_points=[ohlc_data_point, ohlc_data_point],
    )
    prompt_str = prompt.format_prompt()
    assert (
        prompt_str
        == """What is the performance of AAPL on 2021-01-01 00:00:00 with interval W?
    The performance of AAPL on 2021-01-01 00:00:00 (W) is 1.0 2.0 0.5 1.5 10000
What is the performance of AAPL on 2021-01-01 00:00:00 with interval W?
    The performance of AAPL on 2021-01-01 00:00:00 (W) is 1.0 2.0 0.5 1.5 10000"""
    )


def test_create_prompt_text_multiple(
    text_data_point: TextDataPoint, text_template: str
):
    prompt = Prompt(
        template=PromptTemplate(
            input_variables=list(text_data_point.dict_for_template().keys()),
            template=text_template,
        ),
        data_points=[text_data_point, text_data_point],
    )
    prompt_str = prompt.format_prompt()
    assert (
        prompt_str
        == """What is the news for AAPL on 2021-01-01 00:00:00 with interval W?
    The news for AAPL on 2021-01-01 00:00:00 (W) is This is a test
What is the news for AAPL on 2021-01-01 00:00:00 with interval W?
    The news for AAPL on 2021-01-01 00:00:00 (W) is This is a test"""
    )


def test_create_prompt_invalid_data_type(
    ohlc_data_point: OhlcDataPoint, text_data_point: TextDataPoint, text_template: str
):
    with pytest.raises(AssertionError):
        prompt = Prompt(
            template=PromptTemplate(
                input_variables=list(text_data_point.dict_for_template().keys()),
                template=text_template,
            ),
            data_points=[ohlc_data_point],
        )
        _ = prompt.format_prompt()
