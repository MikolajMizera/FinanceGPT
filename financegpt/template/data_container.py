from datetime import datetime
from typing import Generator
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import TypeVar

from financegpt.data.data_point import DataPoint
from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.data.dataset import Dataset
from financegpt.template.templates import SimpleTemplateMeta
from financegpt.template.templates import TemplateMeta

T = TypeVar("T", bound=DataPoint)


class TemplateDataContainer:
    """A container for a template and data points that can be used to create a
    prompt."""

    def __init__(self, template: TemplateMeta, template_data: Sequence[dict[str, str]]):
        self._template = template
        self._template_data = template_data

    def format_prompt(self) -> str:
        assert len(self._template_data), "No data points provided"
        assert all(
            input_var in self._template_data[0]
            for input_var in self._template.input_variables
        ), (
            f"Input variables {self._template.input_variables} must match data "
            f"point keys, provided data point contains {self._template_data[0].keys()}"
        )

        return "\n".join(
            [
                self._template.get_template().format(**data)
                for data in self._template_data
            ]
        )

    def __str__(self) -> str:
        return self.format_prompt()

    def __repr__(self) -> str:
        return repr(self._template_data)


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

    def __repr__(self) -> str:
        return repr(self.containers)

    def __str__(self) -> str:
        return self.format_prompt()

    def append(self, container: TemplateDataContainer) -> None:
        self.containers.append(container)

    def format_prompt(self) -> str:
        return "\n".join([container.format_prompt() for container in self.containers])


class TemplateDataContainerFactory:
    """This factory creates examples of historical data and (optionally) predictions.
    These xamples can be used to create prompts for the model to perform in-context
    training. Creates conatiners form templates and data points from a dataset and
    a template using a sliding window.
    Each container will be created from a window of data points of size
    `window_size` from the dataset.
    """

    def __init__(
        self,
        window_size: int,
        example_template: SimpleTemplateMeta,
        ohlc_template: SimpleTemplateMeta,
        text_template: SimpleTemplateMeta,
    ):
        """Create a prompt factory.

        Args:
        :param window_size: The size of the window to use for creating prompts
        :type window_size: int
        :param example_template: The template to use for creating prompt
        that contains a single example i.e. a collection of `window_size` data
        points
        :type example_template: SimpleTemplateMeta
        :param ohlc_template: The template to use for creating prompts that
        contain a single OHLC data point
        :type ohlc_template: SimpleTemplateMeta
        :param text_template: The template to use for creating prompts that
        contain a single text data point
        :type text_template: SimpleTemplateMeta
        """
        self._window_size = window_size
        self._example_template = example_template
        self._ohlc_template = ohlc_template
        self._text_tempalte = text_template

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def example_template(self) -> SimpleTemplateMeta:
        return self._example_template

    @property
    def ohlc_template(self) -> SimpleTemplateMeta:
        return self._ohlc_template

    @property
    def text_template(self) -> SimpleTemplateMeta:
        return self._text_tempalte

    @staticmethod
    def _skip_weekend(date: datetime) -> int:
        if 0 < 7 - date.weekday() <= 2:
            return 7 - date.weekday()
        else:
            return 0

    def _get_next_window(
        self, index: list[datetime]
    ) -> Generator[list[datetime], None, None]:
        """Get the next window of data points from the dataset. If the window
        ends on a weekend, the window will be extended to the next Monday.
        If there are not enough data points to fill the window, the generator
        will return an empty list.

        Args:
        :param dataset: The dataset to get the next window of data points from
        :type dataset: Dataset[DataPoint]
        :return: The next window of data points
        :rtype: Generator[Dataset[DataPoint], None, None]
        """
        index = sorted(index)
        for windowx_idx in range(len(index) - self._window_size + 1):
            end_i = windowx_idx + self._window_size
            end_i += self._skip_weekend(index[end_i - 1])
            if end_i <= len(index):
                yield index[windowx_idx:end_i]

    def _get_prediction(
        self, dataset: Dataset[OhlcDataPoint]
    ) -> Literal["Increase", "Decrease", ""]:
        """Get a prediction for a window of OHLC data points. The prediction is
        calulated as difference between the close price of the last data point
        and the close price of the first data point in the window.

        Args:
        :param dataset: The dataset to get the predictions from
        :type dataset: Dataset[OhlcDataPoint]
        :return: The prediction
        :rtype: str
        """
        sorted_dataset = sorted(dataset, key=lambda dp: dp.timestamp)
        d_close = sorted_dataset[-1].close - sorted_dataset[0].close
        if d_close > 0:
            return "Increase"
        else:
            return "Decrease"

    def data(
        self,
        ohlc_dataset: Dataset[OhlcDataPoint] | None,
        text_dataset: Dataset[TextDataPoint] | None,
    ) -> TemplateDataContainer:
        ohlc_dataset = ohlc_dataset or Dataset([])
        text_dataset = text_dataset or Dataset([])

        index = self._merge_date_index(ohlc_dataset, text_dataset)

        ohlc_window = Dataset(
            sorted(
                [dp for dp in ohlc_dataset if dp.timestamp in index],
                key=lambda dp: dp.timestamp,
            )
        )
        text_window = Dataset(
            sorted(
                [dp for dp in text_dataset if dp.timestamp in index],
                key=lambda dp: dp.timestamp,
            )
        )

        return self._window_container(ohlc_window, text_window, None)

    def data_windows(
        self,
        ohlc_dataset: Dataset[OhlcDataPoint] | None,
        text_dataset: Dataset[TextDataPoint] | None,
        include_pedictions: bool = False,
    ) -> TemplateDataContainerCollection:
        """Create containers for a template and data points from a dataset using
        a sliding window.
        Each container will be created from a window of data points of size
        `window_size` from the dataset.

        Args:
        :param ohlc_dataset: The dataset to be used to generate examples from,
        if None, the examples will not contain OHLC data points
        :type ohlc_dataset: Dataset[OhlcDataPoint] | None
        :param text_dataset: The dataset to be used to generate examples from,
        if None, the examples will not contain text data points
        :type text_dataset: Dataset[TextDataPoint] | None
        :param include_pedictions: Whether to include predictions in the examples,
        for this to work, the `ohlc_dataset` must be provided
        :type include_pedictions: bool

        """

        examples_collection = TemplateDataContainerCollection([])
        ohlc_dataset = ohlc_dataset or Dataset([])
        text_dataset = text_dataset or Dataset([])

        index = self._merge_date_index(ohlc_dataset, text_dataset)

        for index_window in self._get_next_window(index):
            ohlc_window = Dataset(
                sorted(
                    [dp for dp in ohlc_dataset if dp.timestamp in index_window],
                    key=lambda dp: dp.timestamp,
                )
            )
            text_window = Dataset(
                sorted(
                    [dp for dp in text_dataset if dp.timestamp in index_window],
                    key=lambda dp: dp.timestamp,
                )
            )

            if include_pedictions:
                prediction = self._get_prediction(ohlc_window)
                ohlc_window = ohlc_window[:-1]
                text_window = text_window[:-1]
            else:
                prediction = ""

            example_container = self._window_container(
                ohlc_window, text_window, prediction
            )
            examples_collection.append(example_container)

        return examples_collection

    def _window_container(
        self,
        ohlc_window: Dataset[OhlcDataPoint],
        text_window: Dataset[TextDataPoint],
        prediction: str | None,
    ):
        return TemplateDataContainer(
            template=self._example_template,
            template_data=[
                {
                    "ohlc_window": self._format_prompt_maybe_empty(
                        ohlc_window, self._ohlc_template
                    ),
                    "text_window": self._format_prompt_maybe_empty(
                        text_window, self._text_tempalte
                    ),
                    "prediction": prediction or "",
                }
            ],
        )

    @staticmethod
    def _format_prompt_maybe_empty(dataset_window: Dataset[DataPoint], tempalte) -> str:
        if len(dataset_window):
            return TemplateDataContainer(
                template=tempalte,
                template_data=[
                    dp.dict_for_template(prefix="datapoint_") for dp in dataset_window
                ],
            ).format_prompt()
        else:
            return ""

    @staticmethod
    def _merge_date_index(
        ohlc_dataset: Dataset[OhlcDataPoint], text_dataset: Dataset[TextDataPoint]
    ) -> list[datetime]:
        index = sorted(
            list(
                set([dp.timestamp for dp in ohlc_dataset])
                | set([dp.timestamp for dp in text_dataset])
            )
        )
        return index
