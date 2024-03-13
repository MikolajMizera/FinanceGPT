import os
from datetime import datetime
from datetime import timedelta

from .data_adapter import IntervalType


def get_db_credentials():
    return {
        "host": os.getenv("FINGPT_DB_HOST"),
        "port": int(os.getenv("FINGPT_DB_PORT", 27017)),
        "username": os.getenv("FINGPT_DB_USERNAME"),
        "password": os.getenv("FINGPT_DB_PASSWORD"),
        "db_name": os.getenv("FINGPT_DB_NAME"),
    }


def format_date(date: datetime, interval: IntervalType = "D") -> str:
    if interval in ["H1"]:
        return date.strftime("%Y-%m-%d %H:%M:%S")
    return date.strftime("%Y-%m-%d")


def add_interval(date: datetime, interval: IntervalType) -> datetime:
    if interval == "D":
        return date + timedelta(days=1)
    elif interval == "W":
        return date + timedelta(weeks=1)
    else:
        return date + timedelta(hours=1)
