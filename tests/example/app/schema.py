import random
import uuid
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

ID = "id"
GROUP = "group"
VALUE = "value"
GROUP_COUNT = "group_count"


def create_schema():
    return StructType(
        [
            StructField(ID, StringType()),
            StructField(GROUP, StringType()),
            StructField(VALUE, IntegerType()),
        ]
    )


def create_fake_dataframe(spark: SQLContext, count=1000):
    """Create a dataframe containing some fake data"""
    rows = [
        (
            str(uuid.uuid4()),
            random.choice(["a", "b", "c", "d"]),
            random.randint(10000000, 99999999),
        )
        for i in range(count)
    ]

    return spark.createDataFrame(rows, create_schema())
