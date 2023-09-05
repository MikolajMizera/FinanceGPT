import argparse
import json
import logging
from os import environ

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid

COLLECTIONS = (
    "data_points",
    "inference_results",
    "models",
    "submitted_prompts",
    "users",
)


def main():
    client = MongoClient(
        f"mongodb://{environ['FINGPT_DB_HOST']}:{environ['FINGPT_DB_PORT']}/"
    )
    db = client[environ["FINGPT_DB_NAME"]]

    for collection in COLLECTIONS:
        logging.debug(f"Loading schema for {collection}")
        with open(f"setup/mongo_schemas/{collection}.json", "r") as file:
            schema = json.load(file)
        try:
            db.create_collection(collection, validator=schema)
        except CollectionInvalid:
            logging.debug(f"Collection {collection} already exists, skipping...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup MongoDB collections")
    parser.add_argument("-v", "--verbose", help="Enable verbose logging")
    parser.add_argument("-e", "--env", help="Load environment variables from .env file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.env:
        load_dotenv(dotenv_path=args.env)

    main()
