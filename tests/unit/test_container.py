import pytest
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.template.data_container import TemplateDataContainer
from financegpt.template.templates import ChatTemplateMeta
from financegpt.template.templates import SimpleTemplateMeta


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


@pytest.fixture
def expected_ohlc_chat_prompt():
    return (
        "System: You are a helpful assistant, an expert in finance.\nHuman: What"
        " is the performance of AAPL on 2021-01-01 00:00:00 with interval W?\n"
        "AI: The performance of AAPL on 2021-01-01 00:00:00 (W) is 1.0 2.0 0.5"
        " 1.5 10000"
    )


@pytest.fixture
def expected_text_chat_prompt():
    return (
        "System: You are a helpful assistant, an expert in finance.\nHuman: What"
        " is the news for AAPL on 2021-01-01 00:00:00 with interval W?\n"
        "AI: The news for AAPL on 2021-01-01 00:00:00 (W) is This is a test"
    )


def test_ohlc_template_parsing(
    ohlc_data_point: OhlcDataPoint,
    ohlc_template_meta: SimpleTemplateMeta,
    expected_ohlc_prompt: str,
):
    example_prompt = PromptTemplate(
        input_variables=ohlc_template_meta.input_variables,
        template=ohlc_template_meta.template,
    )

    prompt_str = example_prompt.format(
        **ohlc_data_point.dict_for_template(prefix="datapoint_")
    )
    assert prompt_str == expected_ohlc_prompt


def test_text_template_parsing(
    text_data_point: TextDataPoint,
    text_template_meta: SimpleTemplateMeta,
    expected_text_prompt: str,
):
    example_prompt = PromptTemplate(
        input_variables=text_template_meta.input_variables,
        template=text_template_meta.template,
    )

    prompt_str = example_prompt.format(
        **text_data_point.dict_for_template(prefix="datapoint_")
    )
    assert prompt_str == expected_text_prompt


def test_ohlc_chat_template_parsing(
    ohlc_data_point: OhlcDataPoint,
    ohlc_chat_template_meta: ChatTemplateMeta,
    expected_ohlc_chat_prompt: str,
):
    example_prompt = ChatPromptTemplate.from_messages(ohlc_chat_template_meta.templates)
    prompt_str = example_prompt.format(
        **ohlc_data_point.dict_for_template(prefix="datapoint_")
    )
    assert prompt_str == expected_ohlc_chat_prompt


def test_text_chat_template_parsing(
    text_data_point: TextDataPoint,
    text_chat_template_meta: ChatTemplateMeta,
    expected_text_chat_prompt: str,
):
    example_prompt = ChatPromptTemplate.from_messages(text_chat_template_meta.templates)
    prompt_str = example_prompt.format(
        **text_data_point.dict_for_template(prefix="datapoint_")
    )
    assert prompt_str == expected_text_chat_prompt


def test_create_prompt_ohlc(
    ohlc_data_point: OhlcDataPoint,
    ohlc_template_meta: SimpleTemplateMeta,
    expected_ohlc_prompt: str,
):
    data_point_dict = ohlc_data_point.dict_for_template()
    container = TemplateDataContainer(
        template=ohlc_template_meta,
        template_data=[data_point_dict],
    )
    prompt_str = container.format_prompt()
    assert prompt_str == expected_ohlc_prompt


def test_create_prompt_text(
    text_data_point: TextDataPoint,
    text_template_meta: SimpleTemplateMeta,
    expected_text_prompt: str,
):
    data_point_dict = text_data_point.dict_for_template()
    container = TemplateDataContainer(
        template=text_template_meta,
        template_data=[data_point_dict],
    )
    prompt_str = container.format_prompt()
    assert prompt_str == expected_text_prompt


def test_create_prompt_ohlc_multiple(
    ohlc_data_point: OhlcDataPoint,
    ohlc_template_meta: SimpleTemplateMeta,
    expected_ohlc_prompt: str,
):
    data_point_dict = ohlc_data_point.dict_for_template()
    container = TemplateDataContainer(
        template=ohlc_template_meta,
        template_data=[data_point_dict, data_point_dict],
    )
    prompt_str = container.format_prompt()
    assert prompt_str == (expected_ohlc_prompt + "\n" + expected_ohlc_prompt)


def test_create_prompt_text_multiple(
    text_data_point: TextDataPoint,
    text_template_meta: SimpleTemplateMeta,
    expected_text_prompt: str,
):
    data_point_dict = text_data_point.dict_for_template()
    container = TemplateDataContainer(
        template=text_template_meta,
        template_data=[data_point_dict, data_point_dict],
    )
    prompt_str = container.format_prompt()
    assert prompt_str == (expected_text_prompt + "\n" + expected_text_prompt)


def test_create_prompt_invalid_data_type(
    ohlc_data_point: OhlcDataPoint,
    text_data_point: TextDataPoint,
    text_template_meta: SimpleTemplateMeta,
):
    with pytest.raises(AssertionError):
        prompt = TemplateDataContainer(
            template=SimpleTemplateMeta(
                input_variables=text_template_meta.input_variables,
                template=text_template_meta.template,
                prompt_type="text",
            ),
            template_data=[ohlc_data_point.dict_for_template()],
        )
        _ = prompt.format_prompt()
