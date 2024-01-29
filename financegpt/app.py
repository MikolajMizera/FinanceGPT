import logging
from datetime import datetime
from typing import Any

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
from .template.data_container import TemplateDataContainerFactory
from .template.templates import ChatTemplateMeta
from .template.templates import TemplateMeta


class RequestModel(BaseModel):
    user_msg: str
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

        template_data_containers = self._get_template_data_container(
            request.user_msg,
            request.prediction_symbol,
            request.prediction_end_date,
            requested_historical_data,
        )

        inference_results = self._inference_llm(template_data_containers)
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
        return self._db.get_dataset(symbol, start_date, end_date, interval)

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

    def _get_ohlc_template(self) -> TemplateMeta:
        """
        Returns OHLC template metadata, i.e. input variable names and information
        about OHLC data format.

        :return: OHLC template metadata.
        """
        return self._db.get_templates(filter={"prompt_type": "ohlc"})[0]

    def _get_text_template(self) -> TemplateMeta:
        """
        Returns text template metadata, i.e. input variable names and information
        about text data format.

        :return: Text template metadata.
        """
        return self._db.get_templates(filter={"prompt_type": "text"})[0]

    def _get_system_msg(self) -> str:
        return (
            "You are a helpful AI assistant. You are helping a human to predict"
            "the stock market."
        )

    def _get_template_data_container(
        self,
        user_msg: str,
        prediction_symbol: str,
        prediction_date: datetime,
        dataset: Dataset,
    ) -> TemplateDataContainer:
        """
        Converts dataset and specification of the prompt into template data
        containers.

        :param user_msg: User message.
        :param prediction_symbol: Symbol of a financial instrument.
        :param prediction_date: Date of the prediction, marks the end of the
        prediction window.
        :param dataset: Dataset containing historical data.

        :return: Template data containers, i.e. data that will be used to fill
        in the template along with the template itself.
        """
        container_factory = TemplateDataContainerFactory(window_size=self._window_size)
        ohlc_containers = container_factory.create_containers(
            template=self._get_ohlc_template(),
            dataset=Dataset([d for d in dataset if isinstance(d, OhlcDataPoint)]),
        )
        text_containers = container_factory.create_containers(
            template=self._get_text_template(),
            dataset=Dataset([d for d in dataset if isinstance(d, TextDataPoint)]),
        )

        return TemplateDataContainer(
            template=ChatTemplateMeta(
                input_variables=["prediction_symbol", "prediction_date"],
                prompt_type="text",
                templates=[
                    (
                        "system",
                        f"{self._get_system_msg()}\nExamples:"
                        f"{ohlc_containers.format_prompt()}\n"
                        f"{text_containers.format_prompt()}",
                    ),
                    ("human", user_msg),
                ],
            ),
            template_data=[
                {
                    "prediction_symbol": prediction_symbol,
                    "prediction_date": str(prediction_date),
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
