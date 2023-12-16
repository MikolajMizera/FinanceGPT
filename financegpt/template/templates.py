from abc import abstractmethod
from typing import Literal

from langchain import BasePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel


class TemplateMeta(BaseModel):
    """
    Base class for templates metadata i.e. input variable names and type of the
    data that is contianed in a prompt (OHLC or text)."""

    input_variables: list[str]
    prompt_type: Literal["ohlc", "text"]

    @abstractmethod
    def get_template(self) -> BasePromptTemplate:
        ...


class SimpleTemplateMeta(TemplateMeta):
    """Stores metadata for a simple template that contains only one prompt."""

    template: str

    def get_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=self.input_variables, template=self.template
        )


class ChatTemplateMeta(TemplateMeta):
    """Stores metadata for a chat template that contains multiple prompts
    (`system`, `human`, and `ai`)."""

    templates: list[tuple[Literal["system", "human", "ai"], str]]

    def get_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(self.templates)
