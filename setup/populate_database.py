import argparse
import logging
import os
from datetime import datetime
from typing import Iterable

import yaml
from dotenv import load_dotenv

from financegpt.data.data_adapter import CSVOhlcDataAdapter
from financegpt.data.data_adapter import CSVTextDataAdapter
from financegpt.data.data_connector import MongoDBConnector
from financegpt.data.data_point import DataPoint
from financegpt.data.data_point import IntervalType
from financegpt.data.data_point import OhlcDataPoint
from financegpt.data.data_point import TextDataPoint
from financegpt.data.dataset import Dataset
from financegpt.data.utils import get_db_credentials
from financegpt.template.templates import ChatTemplateMeta
from financegpt.template.templates import SimpleTemplateMeta
from financegpt.template.templates import TemplateMeta

DATA_START_DATE = datetime(2008, 1, 1)
DATA_END_DATE = datetime.today()


def _get_csv_files(data_dir: str) -> list[str]:
    logging.info(f"Reading all CSV files from `{data_dir}` ...")
    return [f for f in os.listdir(data_dir) if f.endswith(".csv")]


def get_dataset_ohlc(
    data_dir: str, symbols: list[str], interval: IntervalType
) -> dict[str, Dataset[OhlcDataPoint]]:
    """
    Returns a list of datasets, one for each symbol in the list. If no symbols are
    i.e. the list is empty, all files in the data folder will be used.
    """
    symbols = _get_csv_files(data_dir) if not symbols else symbols

    symbols = [symbol.split(".")[0] for symbol in symbols]
    ohlc_data_adapter = CSVOhlcDataAdapter(data_dir, index_col="Date")
    return {
        symbol: ohlc_data_adapter.get_dataset(
            symbol, DATA_START_DATE, DATA_END_DATE, interval
        )
        for symbol in symbols
    }


def get_dataset_text(
    data_dir: str, symbols: list[str], interval: IntervalType
) -> dict[str, Dataset[TextDataPoint]]:
    """
    Returns a dataset for the given symbol. If no symbol is provided, all files
    in the data folder will be used.
    """
    symbols = _get_csv_files(data_dir) if not symbols else symbols

    symbols = [symbol.split(".")[0] for symbol in symbols]
    text_data_adapter = CSVTextDataAdapter(
        data_dir, merge_by_interval=True, index_col="Date"
    )
    return {
        symbol: text_data_adapter.get_dataset(
            symbol, DATA_START_DATE, DATA_END_DATE, interval
        )
        for symbol in symbols
    }


def upload_dataset_to_db(db_connection: MongoDBConnector, dataset: Dataset[DataPoint]):
    """
    Uploads the dataset to the database.
    """
    db_connection.store_dataset(dataset)


def get_templates(
    templates_file: str,
) -> tuple[list[SimpleTemplateMeta], list[ChatTemplateMeta]]:
    """
    Returns a list of templates from the given directory.
    """
    logging.info(f"Reading templates from `{templates_file}` ...")
    simple, chat = [], []
    with open(templates_file, "r") as f:
        template_json_files = yaml.safe_load(f)
        for template in template_json_files:
            if "templates" in template:
                chat.append(ChatTemplateMeta(**template))
            else:
                simple.append(SimpleTemplateMeta(**template))
    return simple, chat


def upload_templates_to_db(
    db_connection: MongoDBConnector, templates: Iterable[TemplateMeta]
):
    """
    Uploads the templates to the database.
    """
    db_connection.store_templates(list(templates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate DB with data from CSV files")
    parser.add_argument(
        "-o",
        "--ohlc_data",
        help=(
            "Path to data folder containing CSV files with OHLC data."
            " Each file should be named <symbol>.csv"
        ),
    )
    parser.add_argument(
        "-n",
        "--text_data",
        help=(
            "Path to data folder containing CSV files with news data."
            " Each file should be named <symbol>.csv use UNK.csv for general"
            " news, not associated with a specific stock."
        ),
    )
    parser.add_argument(
        "-t",
        "--prompt_templates",
        help=("Path to a file containing YAML list of prompt templates."),
    )
    parser.add_argument(
        "--ohlc_symbols",
        help=(
            "Comma separated list of symbols to populate the database with."
            " If not provided, all files in the data folder will be used."
        ),
    )
    parser.add_argument(
        "--text_symbols",
        help=(
            "Comma separated list of symbols to populate the database with."
            " If not provided, all files in the data folder will be used."
        ),
    )
    parser.add_argument("-v", "--verbose", help="Enable verbose logging")
    parser.add_argument("-e", "--env", help="Load environment variables from .env file")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.env:
        logging.info(f"Loading environment variables from {args.env}")
        load_dotenv(dotenv_path=args.env)

    ohlc_dataset, text_dataset = None, None
    if args.ohlc_data is not None:
        logging.info(f"Loading OHLC data from {args.ohlc_data}")
        ohlc_dataset = get_dataset_ohlc(args.ohlc_data, args.ohlc_symbols, "D")

    if args.text_data is not None:
        logging.info(f"Loading text data from {args.text_data}")
        text_dataset = get_dataset_text(args.text_data, args.text_symbols, "D")

    simple_templates, chat_templates = None, None
    if args.prompt_templates is not None:
        logging.info(f"Loading simple prompt templates from {args.prompt_templates}")
        simple_templates, chat_templates = get_templates(args.prompt_templates)
    templates = (simple_templates or []) + (chat_templates or [])

    with MongoDBConnector(**get_db_credentials()) as db_connection:
        if ohlc_dataset is not None:
            for dataset_name, dataset in ohlc_dataset.items():
                upload_dataset_to_db(db_connection, dataset)
                logging.info(
                    f"Uploading {dataset_name} dataset of length {len(dataset)} "
                    "to the database..."
                )
        if text_dataset is not None:
            for dataset_name, dataset in text_dataset.items():
                upload_dataset_to_db(db_connection, dataset)
                logging.info(
                    f"Uploading {dataset_name} dataset of length {len(dataset)} "
                    "to the database..."
                )
        if templates:
            logging.info(f"Uploading {len(templates)} templates to the database...")
            upload_templates_to_db(db_connection, templates)
