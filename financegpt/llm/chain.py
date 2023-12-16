from abc import ABC
from abc import abstractmethod

from langchain import OpenAI
from langchain.llms.base import BaseLLM
from langchain.schema.output_parser import StrOutputParser

from ..template.data_container import TemplateDataContainer
from .utils import InferenceResults


class BaseLLMChainInterface(ABC):
    @abstractmethod
    def predict(self, template_data: TemplateDataContainer) -> InferenceResults:
        pass


class LLMChainInterface(BaseLLMChainInterface):
    def __init__(self, llm: BaseLLM):
        self._chain = llm | StrOutputParser()

    def predict(self, template_data: TemplateDataContainer) -> InferenceResults:
        return InferenceResults(
            output=self._chain.invoke(template_data.format_prompt())
        )


AvaiableOpenAIModels = ("gpt-3.5-turbo-instruct", "gpt-4-turbo")


class LLMChainInterfaceFactory:
    @staticmethod
    def _create_openai_llm(llm_type: str, **kwargs) -> OpenAI:
        return OpenAI(model=llm_type, **kwargs)

    @staticmethod
    def create_llm_chain(llm_type: str, **kwargs) -> LLMChainInterface:
        if llm_type in AvaiableOpenAIModels:
            llm = LLMChainInterfaceFactory._create_openai_llm(llm_type, **kwargs)
        else:
            raise ValueError(f"LLM type {llm_type} not supported.")
        return LLMChainInterface(llm)
