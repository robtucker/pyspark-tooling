import os
import random
import uuid
from pyspark.sql import SQLContext

from app.schema import schema
from app.group import group_rows

from grada_pyspark_utils.spark import get_default_conf
from grada_pyspark_utils.io import write_parquet
from grada_logger import log


def main():
    "A minimal application designed to test deploying pyspark jobs to EMR"

    parallellism = os.environ["PYSPARK_PARALLELLISM"]
    spark = get_default_conf("grada_pyspark_test", parallellism)

    df = create_fake_dataframe(spark)
    log.info("created fake dataframe", record_count=df.count())

    # check our zipped code bundle is working
    res = group_rows(df)

    path = os.environ["INTEGRATION_TEST_TARGET"]

    log.info("writing dataframe to s3", path=path)

    write_parquet(path, res)


def create_fake_dataframe(spark: SQLContext, count=1000):
    """Create a dataframe containing some fake data"""
    rows = [
        (
            str(uuid.uuid4()),
            random.choice(["a", "b", "c", "d"]),
            random.uniform(10000000, 99999999),
        )
        for i in range(count)
    ]

    return spark.createDataFrame(rows, schema())
