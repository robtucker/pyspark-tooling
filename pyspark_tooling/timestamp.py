import pytz
from datetime import datetime

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

# for use with datetime.strftime
PYTHON_DATE_FORMAT = "%Y-%m-%d"
PYTHON_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

# for use with pyspark sql format_date function
PYSPARK_DATE_FORMAT = "yyyy-MM-dd"
PYSPARK_TIME_FORMAT = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"


def with_timestamp(col_name: str, timestamp: datetime, df: DataFrame) -> DataFrame:
    """Add a timestamp literal to each row"""
    # first convert from python datetime into utc timestamp
    t = format_timestamp(timestamp)
    # then convert from string into timestamp type
    return df.withColumn(col_name, F.lit(t))


def utcnow():
    """The native python utcnow is dangerous as it has no tzinfo
    and therefore cannot be converted into different timezones reliably.
    This is easily fixed by replacing the tzinfo."""
    return datetime.utcnow().replace(tzinfo=pytz.utc)


def format_timestamp(t: datetime) -> str:
    """Format a timestamp in the correct format"""
    # if a timezone has been set and it isn't utc
    if t.tzinfo is not None and t.tzinfo != pytz.utc:
        # ensure the datetime is converted to utc before being rendered
        t = t.astimezone(tz=pytz.utc)
    # otherwise we can only hope that the user of this function is using utc
    return t.strftime(PYTHON_TIME_FORMAT)[:-4] + "Z"
    # return t.isoformat(sep="T", timespec="milliseconds") + "Z"
