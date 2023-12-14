from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from pymongo import MongoClient

from ..prompting.prompt import TemplateData
from .data_adapter import DataAdapter
from .data_point import DataPoint
from .data_point import IntervalType
from .data_point import OhlcDataPoint
from .data_point import TextDataPoint
from .dataset import Dataset

DATA_COLLECTION = "data"
TEMPLATES_COLLECTION = "templates"


class DBConnector(DataAdapter[DataPoint], ABC):
    """
    An interface for data connectors. Data connectors are responsible for
    storing and retrieving data from a data source. Retrieving data is done
    through the get_data method, which is a part of the DataAdapter interface.
    """

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError

    @abstractmethod
    def store_dataset(self, dataset: Dataset[DataPoint]):
        raise NotImplementedError

    @abstractmethod
    def store_templates(self, templates: list[TemplateData]):
        raise NotImplementedError

    @abstractmethod
    def get_templates(self, filter: dict[str, str] | None = None) -> list[TemplateData]:
        raise NotImplementedError


class MongoDBConnector(DBConnector):
    def __init__(
        self, username: str, password: str, host: str, port: int, db_name: str
    ):
        self._client = MongoClient(
            username=username,
            password=password,
            host=host,
            port=port,
        )
        self._db_name = db_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._client.close()

    def store_dataset(self, dataset: Dataset[DataPoint]):
        for data_point in dataset:
            self._client[self._db_name][DATA_COLLECTION].insert_one(
                data_point.model_dump()
            )

    def _convert_ohlc_data_points(
        self, data_points: list[dict[str, Any]]
    ) -> list[OhlcDataPoint]:
        return [OhlcDataPoint(**data_point) for data_point in data_points]

    def _convert_text_data_points(
        self, data_points: list[dict[str, Any]]
    ) -> list[TextDataPoint]:
        return [TextDataPoint(**data_point) for data_point in data_points]

    def get_dataset(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: IntervalType | None,
    ) -> Dataset[OhlcDataPoint] | Dataset[TextDataPoint]:
        data = list(
            self._client[self._db_name][DATA_COLLECTION].find(
                {
                    "symbol": symbol,
                    "timestamp": {
                        "$gte": start_date,
                        "$lte": end_date,
                    },
                    "interval": interval,
                },
                **self.kwargs,
            )
        )

        try:
            return Dataset(data=self._convert_ohlc_data_points(data))
        except ValidationError:
            return Dataset(data=self._convert_text_data_points(data))

    def store_templates(self, templates: list[TemplateData]):
        for template in templates:
            self._client[self._db_name][TEMPLATES_COLLECTION].insert_one(
                template.model_dump()
            )

    def get_templates(self, filter: dict[str, str] | None = None) -> list[TemplateData]:
        return [
            TemplateData(**template)
            for template in self._client[self._db_name][TEMPLATES_COLLECTION].find(
                filter, projection={"_id": False}
            )
        ]
