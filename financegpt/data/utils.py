from datetime import datetime
from datetime import timedelta

from .data_adapter import IntervalType


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
