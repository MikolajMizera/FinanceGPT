from typing import Generator
from typing import Iterator

from financegpt.data.data_point import DataPoint
from financegpt.data.dataset import Dataset
from financegpt.template.templates import TemplateMeta


class TemplateDataContainer:
    """A container for a template and data points that can be used to create a
    prompt."""

    def __init__(self, template: TemplateMeta, template_data: list[dict[str, str]]):
        self._template = template
        self._template_data = template_data

    def format_prompt(self) -> str:
        assert all(
            input_var in self._template_data[0]
            for input_var in self._template.input_variables
        ), "Input variables must match data point keys, provided data point is "
        f"contains {self._template_data}"

        return "\n".join(
            [
                self._template.get_template().format(**data)
                for data in self._template_data
            ]
        )

    def __str__(self) -> str:
        return self.format_prompt()


class TemplateDataContainerCollection:
    def __init__(self, containers: list[TemplateDataContainer]):
        self._containers = containers

    @property
    def containers(self) -> list[TemplateDataContainer]:
        return self._containers

    def __iter__(self) -> Iterator[TemplateDataContainer]:
        return iter(self._containers)

    def __len__(self) -> int:
        return len(self._containers)

    def __getitem__(self, index: int) -> TemplateDataContainer:
        return self._containers[index]

    def __add__(
        self, other: "TemplateDataContainerCollection"
    ) -> "TemplateDataContainerCollection":
        return TemplateDataContainerCollection(self.containers + other.containers)

    def format_prompt(self) -> str:
        return "\n".join([container.format_prompt() for container in self.containers])


class TemplateDataContainerFactory:
    """Creates conatiners form templates and data points from a dataset and a
    template using a sliding window.
    Each container will be created from a window of data points of size
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

    def create_containers(
        self, template: TemplateMeta, dataset: Dataset
    ) -> TemplateDataContainerCollection:
        """Create containers for a template and data points from a dataset using
        a sliding window.
        Each container will be created from a window of data points of size
        `window_size` from the dataset.

        Args:
        :param template: The template to use for prompts
        :type template: PromptTemplate
        :param dataset: The dataset to create prompts from
        :type dataset: Dataset
        :return: A list of prompts
        :rtype: list[Prompt]
        """
        return TemplateDataContainerCollection(
            [
                TemplateDataContainer(
                    template=template,
                    template_data=[
                        dp.dict_for_template(prefix="datapoint_")
                        for dp in data_window.data
                    ],
                )
                for data_window in self._get_next_data_points(dataset)
            ]
        )
