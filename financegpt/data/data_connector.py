from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Iterable

from pymongo import MongoClient

from ..template.templates import TemplateMeta
from ..template.templates import TemplateMetaFactory
from .data_adapter import DataAdapter
from .data_point import DataPoint
from .data_point import DataPointFactory
from .data_point import IntervalType
from .dataset import Dataset

DATA_COLLECTION = "data"
TEMPLATES_COLLECTION = "templates"


class DBConnector(DataAdapter[DataPoint], ABC):
    """
    An interface for data connectors. Data connectors are responsible for
    storing and retrieving data from a data source. Retrieving data is done
    through the get_data method, which is a part of the DataAdapter interface.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def store_dataset(self, dataset: Dataset[DataPoint]):
        raise NotImplementedError

    @abstractmethod
    def store_templates(self, templates: list[TemplateMeta]):
        raise NotImplementedError

    @abstractmethod
    def get_templates(self, filter: dict[str, str] | None = None) -> list[TemplateMeta]:
        raise NotImplementedError


class MongoDBConnector(DBConnector):
    def __init__(
        self, username: str, password: str, host: str, port: int, db_name: str, **kwargs
    ):
        super().__init__(**kwargs)
        self._client = MongoClient(
            username=username,
            password=password,
            host=host,
            port=port,
        )
        self._db_name = db_name

    def __enter__(self):
        return self

    def close(self):
        self._client.close()

    def store_dataset(self, dataset: Dataset[DataPoint]):
        self._client[self._db_name][DATA_COLLECTION].insert_many(
            (data_point.model_dump() for data_point in dataset.data)
        )

    def _parse_datapoint(self, datapoints: Iterable[dict]) -> list[DataPoint]:
        return [
            DataPointFactory.create_data_point(**datapoint) for datapoint in datapoints
        ]

    def get_dataset(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: IntervalType | None,
    ) -> Dataset[DataPoint]:
        data = self._client[self._db_name][DATA_COLLECTION].find(
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
        return Dataset(data=self._parse_datapoint(data))

    def store_templates(self, templates: list[TemplateMeta]):
        for template in templates:
            self._client[self._db_name][TEMPLATES_COLLECTION].insert_one(
                template.model_dump()
            )

    def _parse_templates(self, templates: Iterable[dict]) -> list[TemplateMeta]:
        return [
            TemplateMetaFactory.create_tempate_meta(tempalte) for tempalte in templates
        ]

    def get_templates(self, filter: dict[str, str] | None = None) -> list[TemplateMeta]:
        return self._parse_templates(
            self._client[self._db_name][TEMPLATES_COLLECTION].find(
                filter, projection={"_id": False}
            )
        )
