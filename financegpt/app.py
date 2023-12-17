from datetime import datetime
from typing import Any

from pydantic import BaseModel

from .data.data_connector import MongoDBConnector
from .data.data_point import DataPoint
from .data.data_point import IntervalType
from .data.dataset import Dataset
from .llm.chain import LLMChainInterfaceFactory
from financegpt.llm.utils import InferenceResults
from financegpt.template.data_container import TemplateDataContainer
from financegpt.template.data_container import TemplateDataContainerFactory
from financegpt.template.templates import ChatTemplateMeta
from financegpt.template.templates import TemplateMeta


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
        **kwargs,
    ):
        self._llm_chain = LLMChainInterfaceFactory.create_llm_chain(llm_model)
        self._db = MongoDBConnector(**connection_kwargs)
        self._window_size = window_size

    def __del__(self):
        self._db.close()

    def process_request(self, request: RequestModel) -> str:
        """
        Processes request and returns response.
        """
        requested_historical_data = self._get_requested_data(
            symbol=request.prediction_symbol,
            start_date=request.historical_data_start_date,
            end_date=request.historical_data_end_date,
            interval=request.historical_data_interval,
        )

        template_data_containers = self._get_template_data_container(
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
        Returns requested data.
        """
        return self._db.get_dataset(symbol, start_date, end_date, interval)

    def _get_ohlc_template(self) -> TemplateMeta:
        """
        Returns OHLC template.
        """
        return self._db.get_templates(filter={"type": "ohlc"})[0]

    def _get_text_template(self) -> TemplateMeta:
        """
        Returns text template.
        """
        return self._db.get_templates(filter={"type": "text"})[0]

    def _get_system_msg(self) -> str:
        raise NotImplementedError

    def _get_user_msg_template(self) -> str:
        raise NotImplementedError

    def _get_template_data_container(
        self, prediction_symbol: str, prediction_date: datetime, dataset: Dataset
    ) -> TemplateDataContainer:
        """
        Returns template data containers.
        """
        container_factory = TemplateDataContainerFactory(window_size=self._window_size)
        ohlc_containers = container_factory.create_containers(
            template=self._get_ohlc_template(), dataset=dataset
        )
        text_containers = container_factory.create_containers(
            template=self._get_text_template(), dataset=dataset
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
                    ("human", self._get_user_msg_template()),
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
        return self._llm_chain.predict(template_data)

    def _parse_results(self, inference_results: InferenceResults) -> str:
        if inference_results.error_code == 0:
            return inference_results.output
        return self._handle_error(inference_results.error_code)

    def _handle_error(self, error_code: int) -> str:
        return f"Error {error_code} occurred!"
