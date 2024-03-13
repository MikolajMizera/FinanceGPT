import logging
from datetime import datetime
from typing import Any
from typing import Sequence

from pydantic import BaseModel

from .data.data_connector import MongoDBConnector
from .data.data_point import DataPoint
from .data.data_point import IntervalType
from .data.data_point import OhlcDataPoint
from .data.data_point import TextDataPoint
from .data.dataset import Dataset
from .llm.chain import LLMChainInterfaceFactory
from .llm.utils import InferenceResults
from .template.data_container import TemplateDataContainer
from .template.data_container import TemplateDataContainerCollection
from .template.data_container import TemplateDataContainerFactory
from .template.templates import ChatTemplateMeta
from .template.templates import SimpleTemplateMeta


class RequestModel(BaseModel):
    historical_data_start_date: datetime
    historical_data_end_date: datetime
    historical_data_interval: IntervalType
    prediction_symbol: str
    prediction_end_date: datetime


class AppController:
    def __init__(
        self,
        llm_model: str,
        connection_kwargs: dict[str, Any],
        window_size: int,
    ):
        """
        Initializes AppController. This class is responsible for processing
        requests and returning responses from LLM.

        :param llm_model: LLM model name. See:
         - `financegpt.llm.AvaiableOpenAIModels`
        for supported models.
        :param connection_kwargs: MongoDB connection kwargs.
        :param window_size: Window size for LLM.
        """

        logging.info("Initializing AppController...")

        logging.info(f"Initializing LLM model: {llm_model}")
        self._llm_chain = LLMChainInterfaceFactory.create_llm_chain(llm_model)
        logging.info("Establishing connection to MongoDB...")
        self._db = MongoDBConnector(**connection_kwargs)
        logging.info(f"Example prompt data window size: {window_size}")
        self._window_size = window_size

        self._container_factory = TemplateDataContainerFactory(
            window_size=self._window_size,
            example_template=self._get_simple_template(type="example"),
            ohlc_template=self._get_simple_template(type="ohlc"),
            text_template=self._get_simple_template(type="text"),
        )

    def __del__(self):
        self._db.close()

    def process_request(self, request: RequestModel) -> str:
        """
        Processes users request and returns response from LLM.

        :param request: A `RequestModel` instance containing data for LLM prompt.
        :return: Text response from LLM.
        """

        logging.info("Processing user request...")
        requested_historical_data = self._get_requested_data(
            symbol=request.prediction_symbol,
            start_date=request.historical_data_start_date,
            end_date=request.historical_data_end_date,
            interval=request.historical_data_interval,
        )
        self._check_dataset_length(requested_historical_data)
        ohlc_historical_data = self._filter_ohlc_dataset(requested_historical_data)
        text_historical_data = self._filter_text_dataset(requested_historical_data)

        system_container = self._get_system_data_container(
            examples=self._get_data_windows(
                ohlc_dataset=ohlc_historical_data,
                text_dataset=text_historical_data,
                include_predictions=True,
            )
        )
        user_container = self._get_user_request_data_container(
            historical_data=self._container_factory.data(
                ohlc_dataset=ohlc_historical_data, text_dataset=text_historical_data
            ),
            symbol=request.prediction_symbol,
            prediction_end_date=request.prediction_end_date,
        )
        chat_request = self._get_chat_request(
            system_container=system_container, user_container=user_container
        )

        inference_results = self._inference_llm(chat_request)
        response = self._parse_results(inference_results)
        return response

    def _get_requested_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: IntervalType,
    ) -> Dataset[DataPoint]:
        """
        Query database for requested data.

        :param symbol: Symbol of a financial instrument.
        :param start_date: Start date of the requested data.
        :param end_date: End date of the requested data.
        :param interval: Interval of the requested data.
        :return: Requested data.
        """
        return self._db.get_dataset(
            symbol, start_date, end_date, interval
        ) + self._db.get_dataset("UNK", start_date, end_date, interval)

    def _check_dataset_length(self, dataset: Dataset[DataPoint]) -> None:
        """
        Checks if dataset is long enough to be used in LLM.

        :param dataset: Dataset containing historical data.
        """
        if len(dataset) < self._window_size:
            raise ValueError(
                f"Dataset length is {len(dataset)}, but must be at least "
                f"{self._window_size}!"
            )

    def _filter_ohlc_dataset(
        self, dataset: Dataset[DataPoint]
    ) -> Dataset[OhlcDataPoint]:
        return Dataset([d for d in dataset if isinstance(d, OhlcDataPoint)])

    def _filter_text_dataset(
        self, dataset: Dataset[DataPoint]
    ) -> Dataset[TextDataPoint]:
        return Dataset([d for d in dataset if isinstance(d, TextDataPoint)])

    def _select_simple_template(
        self, templates: Sequence[SimpleTemplateMeta]
    ) -> SimpleTemplateMeta:
        """
        Selects a simple template from a list of templates. Currently picks the
        first one, but can be extended to use more sophisticated logic.

        :param templates: List of simple templates.
        :return: Simple template.
        """
        try:
            return templates[0]
        except IndexError:
            raise ValueError("No instances of `SimpleTemplateMeta` found in db!")

    def _get_simple_template(self, type: str) -> SimpleTemplateMeta:
        """
        Get a simple template by type from the database.

        : param type: Type of the template.
        : return: Simple template.
        """
        simple_templates = [
            t
            for t in self._db.get_templates(filter={"prompt_type": type})
            if isinstance(t, SimpleTemplateMeta)
        ]
        template = self._select_simple_template(simple_templates)
        return template

    def _get_chat_template(self, type: str) -> ChatTemplateMeta:
        """
        Get a chat template by type from the database.

        : param type: Type of the template.
        : return: Chat template.
        """
        template = self._db.get_templates(filter={"prompt_type": type})[0]
        assert isinstance(template, ChatTemplateMeta)
        return template

    def _get_data_windows(
        self,
        ohlc_dataset: Dataset[OhlcDataPoint],
        text_dataset: Dataset[TextDataPoint],
        include_predictions=False,
    ) -> TemplateDataContainerCollection:
        """
        Get data windows from the datasets.

        :param ohlc_dataset: Dataset containing OHLC data.
        :param text_dataset: Dataset containing text data.
        :param include_predictions: Include predictions in the data windows.
        :return: Data windows.
        """

        return self._container_factory.data_windows(
            ohlc_dataset=ohlc_dataset,
            text_dataset=text_dataset,
            include_pedictions=include_predictions,
        )

    def _get_system_data_container(
        self, examples: TemplateDataContainerCollection
    ) -> TemplateDataContainer:
        """
        Converts dataset and specification of the prompt into template data
        containers.

        :a
        :return: Template data containers, i.e. data that will be used to fill
        in the template along with the template itself.
        """

        return TemplateDataContainer(
            template=self._get_simple_template(type="system"),
            template_data=[
                {
                    "ohlc_format": self._container_factory.ohlc_template.template,
                    "text_format": self._container_factory.text_template.template,
                    "examples": examples.format_prompt(),
                }
            ],
        )

    def _get_user_request_data_container(
        self,
        historical_data: TemplateDataContainer,
        symbol: str,
        prediction_end_date: datetime,
    ) -> TemplateDataContainer:
        return TemplateDataContainer(
            template=self._get_simple_template(type="user_request"),
            template_data=[
                {
                    "user_request_data": historical_data.format_prompt(),
                    "symbol": symbol,
                    "prediction_end_date": prediction_end_date.isoformat(),
                }
            ],
        )

    def _get_chat_request(
        self,
        system_container: TemplateDataContainer,
        user_container: TemplateDataContainer,
    ) -> TemplateDataContainer:
        """
        Converts user message into template data containers.

        :param user_msg: User message.
        :return: Template data containers, i.e. data that will be used to fill
        in the template along with the template itself.
        """
        chat_template = self._get_chat_template(type="request")
        return TemplateDataContainer(
            template=chat_template,
            template_data=[
                {
                    "system": system_container.format_prompt(),
                    "user_request": user_container.format_prompt(),
                }
            ],
        )

    def _inference_llm(self, template_data: TemplateDataContainer) -> InferenceResults:
        """
        Performs inference using LLM.

        :param template_data: Template data containers, i.e. data that will be used
        to fill in the template along with the template itself.
        :return: Inference results.
        """
        return self._llm_chain.predict(template_data)

    def _parse_results(self, inference_results: InferenceResults) -> str:
        """
        Parses inference results, checks for errors.

        :param inference_results: Inference results.
        :return: Parsed results.
        """
        if inference_results.error_code == 0:
            return inference_results.output
        return self._handle_error(inference_results.error_code)

    def _handle_error(self, error_code: int) -> str:
        """
        Basic error handler.
        """
        return f"Error {error_code} occurred!"
