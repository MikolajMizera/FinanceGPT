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
    def __init__(
        self,
        ohlc_template: PromptTemplate,
        text_template: PromptTemplate,
        window_size: int,
    ):
        self._ohlc_template = ohlc_template
        self._text_template = text_template
        self._window_size = window_size

    def _get_next_data_points(
        self, dataset: Dataset[DataPoint]
    ) -> Generator[Dataset[DataPoint], None, None]:
        for windowx_idx in range(0, len(dataset), self._window_size):
            window = dataset[windowx_idx : windowx_idx + self._window_size]
            if isinstance(window, DataPoint):
                window = Dataset(data=[window])
            yield window

    def create_prompts(
        self, template: PromptTemplate, dataset: Dataset
    ) -> list[Prompt]:
        return [
            Prompt(template=template, data_points=data_window.data)
            for data_window in self._get_next_data_points(dataset)
        ]
