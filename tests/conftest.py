import os
from datetime import datetime
from datetime import timedelta
from typing import Any

import pytest
import yaml

from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.data.dataset import Dataset
from financegpt.template.data_container import TemplateDataContainerFactory
from financegpt.template.templates import ChatTemplateMeta
from financegpt.template.templates import SimpleTemplateMeta


@pytest.fixture
def ohlc_data_point():
    return OhlcDataPoint(
        symbol="AAPL",
        timestamp=datetime(2021, 1, 1),
        interval="W",
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=10000,
    )


@pytest.fixture
def text_data_point():
    return TextDataPoint(
        symbol="AAPL",
        timestamp=datetime(2021, 1, 1),
        interval="W",
        text="This is a test",
    )


@pytest.fixture
def templates() -> dict[str, SimpleTemplateMeta | ChatTemplateMeta]:
    with open(f"{os.path.dirname(__file__)}/test_data/templates.yaml", "r") as fh:
        return {
            template["prompt_type"]: ChatTemplateMeta(**template)
            if "templates" in template
            else SimpleTemplateMeta(**template)
            for template in yaml.safe_load(fh)
        }


@pytest.fixture
def system_msg() -> str:
    return "You are a helpful assistant, an expert in finance."


@pytest.fixture
def human_msg_ohlc() -> str:
    return (
        "What is the performance of {datapoint_symbol} on {datapoint_timestamp}"
        " with interval {datapoint_interval}?"
    )


@pytest.fixture
def ai_msg_ohlc() -> str:
    return (
        "The performance of {datapoint_symbol} on {datapoint_timestamp}"
        " ({datapoint_interval}) is {datapoint_open} {datapoint_high} {datapoint_low}"
        " {datapoint_close} {datapoint_volume}"
    )


@pytest.fixture
def human_msg_text() -> str:
    return (
        "What is the news for {datapoint_symbol} on {datapoint_timestamp}"
        " with interval {datapoint_interval}?"
    )


@pytest.fixture
def ai_msg_text() -> str:
    return (
        "The news for {datapoint_symbol} on {datapoint_timestamp}"
        " ({datapoint_interval}) is {datapoint_text}"
    )


@pytest.fixture
def ohlc_template_meta(human_msg_ohlc: str, ai_msg_ohlc: str) -> SimpleTemplateMeta:
    return SimpleTemplateMeta(
        input_variables=[
            "datapoint_symbol",
            "datapoint_timestamp",
            "datapoint_interval",
            "datapoint_open",
            "datapoint_high",
            "datapoint_low",
            "datapoint_close",
            "datapoint_volume",
        ],
        template=f"{human_msg_ohlc}\n{ai_msg_ohlc}",
        prompt_type="ohlc",
    )


@pytest.fixture
def ohlc_chat_template_meta(
    system_msg: str, human_msg_ohlc: str, ai_msg_ohlc: str
) -> ChatTemplateMeta:
    return ChatTemplateMeta(
        input_variables=[
            "datapoint_symbol",
            "datapoint_timestamp",
            "datapoint_interval",
            "datapoint_open",
            "datapoint_high",
            "datapoint_low",
            "datapoint_close",
            "datapoint_volume",
        ],
        templates=[
            ("system", system_msg),
            ("human", human_msg_ohlc),
            ("ai", ai_msg_ohlc),
        ],
        prompt_type="ohlc",
    )


@pytest.fixture
def text_template_meta(human_msg_text: str, ai_msg_text: str) -> SimpleTemplateMeta:
    return SimpleTemplateMeta(
        input_variables=[
            "datapoint_symbol",
            "datapoint_timestamp",
            "datapoint_interval",
            "datapoint_text",
        ],
        template=f"{human_msg_text}\n{ai_msg_text}",
        prompt_type="text",
    )


@pytest.fixture
def text_chat_template_meta() -> ChatTemplateMeta:
    return ChatTemplateMeta(
        input_variables=[
            "datapoint_symbol",
            "datapoint_timestamp",
            "datapoint_interval",
            "datapoint_text",
        ],
        templates=[
            ("system", "You are a helpful assistant, an expert in finance."),
            (
                "human",
                "What is the news for {datapoint_symbol} on "
                "{datapoint_timestamp} with interval {datapoint_interval}?",
            ),
            (
                "ai",
                "The news for {datapoint_symbol} on {datapoint_timestamp}"
                " ({datapoint_interval}) is {datapoint_text}",
            ),
        ],
        prompt_type="text",
    )


@pytest.fixture
def example_template_meta():
    return SimpleTemplateMeta(
        input_variables=["ohlc_window", "text_window", "prediction"],
        prompt_type="example",
        template=(
            "Price data:\n{ohlc_window}\nMarket News:\n{text_window}\n"
            "Prediction: {prediction}"
        ),
    )


@pytest.fixture
def ohlc_dataset_5days_weekend():
    return Dataset(
        data=[
            OhlcDataPoint(
                open=1.0,
                high=2.0,
                low=0.5,
                close=1.5,
                volume=10000,
                timestamp=datetime(2021, 1, 1) + timedelta(days=i),
                symbol="AAPL",
                interval="W",
            )
            for i in range(5)
        ]
    )


@pytest.fixture
def ohlc_dataset_5days_weekday():
    return Dataset(
        data=[
            OhlcDataPoint(
                open=1.0,
                high=2.0,
                low=0.5,
                close=1.5,
                volume=10000,
                timestamp=datetime(2021, 1, 4) + timedelta(days=i),
                symbol="AAPL",
                interval="W",
            )
            for i in range(5)
        ]
    )


@pytest.fixture
def container_factory_2d(templates: dict):
    return TemplateDataContainerFactory(
        window_size=2,
        example_template=templates["example"],
        ohlc_template=templates["ohlc"],
        text_template=templates["text"],
    )


@pytest.fixture
def example_request() -> dict[str, Any]:
    return {
        "user_msg": "What will the AAPL do tomorrow?",
        "prediction_symbol": "AAPL",
        "prediction_end_date": datetime(2023, 12, 31),
        "historical_data_start_date": datetime(2021, 1, 1),
        "historical_data_end_date": datetime(2021, 12, 17),
        "historical_data_interval": "W",
    }
