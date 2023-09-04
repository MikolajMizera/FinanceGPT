from typing import Generator

from langchain.prompts.prompt import PromptTemplate

from financegpt.data.data_point import DataPoint
from financegpt.data.dataset import Dataset


class Prompt:
    def __init__(self, template: PromptTemplate, data_points: list[DataPoint]):
        self._template = template
        self._data_points = data_points

    def format_prompt(self) -> str:
        assert all(
            input_var in self._data_points[0].dict_for_template()
            for input_var in self._template.input_variables
        ), "Input variables must match data point keys, provided data point is "
        f"of type {type(self._data_points[0])}"

        return "\n".join(
            [
                self._template.format(
                    **data_point.dict_for_template(prefix="datapoint_")
                )
                for data_point in self._data_points
            ]
        )


class PromptFactory:
    """Creates prompts from a dataset and a template using a sliding window.
    Each prompt will be created from a window of data points of size
    `window_size` from the dataset.
    """

    def __init__(
        self,
        window_size: int,
    ):
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
            Prompt(template=template, data_points=data_window.data)
            for data_window in self._get_next_data_points(dataset)
        ]


class PromptCollection:
    def __init__(self, prompts: list[Prompt]):
        self._prompts = prompts

    @property
    def prompts(self) -> list[Prompt]:
        return self._prompts

    def __iter__(self):
        return iter(self._prompts)

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, index):
        return self._prompts[index]

    def __add__(self, other: "PromptCollection") -> "PromptCollection":
        return PromptCollection(self.prompts + other.prompts)

    def format_prompt(self) -> str:
        return "\n".join([prompt.format_prompt() for prompt in self.prompts])
