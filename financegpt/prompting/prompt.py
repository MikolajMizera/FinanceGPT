from typing import Generator
from typing import Iterator
from typing import Literal

from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel

from financegpt.data.data_point import DataPoint
from financegpt.data.dataset import Dataset


class TemplateData(BaseModel):
    input_variables: list[str]
    template: str
    prompt_type: Literal["ohlc", "text"]

    def get_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=self.input_variables, template=self.template
        )


class Prompt:
    def __init__(self, template: PromptTemplate, template_data: list[dict[str, str]]):
        self._template = template
        self._template_data = template_data

    def format_prompt(self) -> str:
        assert all(
            input_var in self._template_data[0]
            for input_var in self._template.input_variables
        ), "Input variables must match data point keys, provided data point is "
        f"contains {self._template_data}"

        return "\n".join(
            [self._template.format(**data) for data in self._template_data]
        )

    def __str__(self) -> str:
        return self.format_prompt()


class PromptFactory:
    """Creates prompts from a dataset and a template using a sliding window.
    Each prompt will be created from a window of data points of size
    `window_size` from the dataset.
    """

    def __init__(self, window_size: int):
        """Create a prompt factory.

        Args:
        :param window_size: The size of the sliding window to use
        :type window_size: int
        """
        self._window_size = window_size

    def _get_next_data_points(
        self, dataset: Dataset[DataPoint]
    ) -> Generator[Dataset[DataPoint], None, None]:
        """Get the next window of data points from the dataset.

        Args:
        :param dataset: The dataset to get the next window of data points from
        :type dataset: Dataset[DataPoint]
        :return: The next window of data points
        :rtype: Generator[Dataset[DataPoint], None, None]
        """
        for windowx_idx in range(len(dataset) - self._window_size + 1):
            yield dataset[windowx_idx : windowx_idx + self._window_size]

    def create_prompts(
        self, template: PromptTemplate, dataset: Dataset
    ) -> list[Prompt]:
        """Create prompts from a dataset and a template using a sliding window.
        Each prompt will be created from a window of data points of size
        `window_size` from the dataset.

        Args:
        :param template: The template to use for prompts
        :type template: PromptTemplate
        :param dataset: The dataset to create prompts from
        :type dataset: Dataset
        :return: A list of prompts
        :rtype: list[Prompt]
        """
        return [
            Prompt(
                template=template,
                template_data=[
                    dp.dict_for_template(prefix="datapoint_") for dp in data_window.data
                ],
            )
            for data_window in self._get_next_data_points(dataset)
        ]


class PromptCollection:
    def __init__(self, prompts: list[Prompt]):
        self._prompts = prompts

    @property
    def prompts(self) -> list[Prompt]:
        return self._prompts

    def __iter__(self) -> Iterator[Prompt]:
        return iter(self._prompts)

    def __len__(self) -> int:
        return len(self._prompts)

    def __getitem__(self, index):
        return self._prompts[index]

    def __add__(self, other: "PromptCollection") -> "PromptCollection":
        return PromptCollection(self.prompts + other.prompts)

    def format_prompt(self) -> str:
        return "\n".join([prompt.format_prompt() for prompt in self.prompts])
