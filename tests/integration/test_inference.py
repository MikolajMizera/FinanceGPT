import pytest

from financegpt.data.data_point import TextDataPoint
from financegpt.llm.chain import LLMChainInterface
from financegpt.llm.chain import LLMChainInterfaceFactory
from financegpt.template.data_container import TemplateDataContainer
from financegpt.template.templates import ChatTemplateMeta


@pytest.fixture
def llm_chain():
    return LLMChainInterfaceFactory.create_llm_chain("gpt-3.5-turbo-instruct")


def test_inference(
    llm_chain: LLMChainInterface,
    text_chat_inference_meta: ChatTemplateMeta,
    text_data_point: TextDataPoint,
):
    template_data_container = TemplateDataContainer(
        template=text_chat_inference_meta,
        template_data=[text_data_point.dict_for_template()],
    )
    response = llm_chain.predict(template_data_container)
    assert len(response.output)
