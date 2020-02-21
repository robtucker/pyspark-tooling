from app.schema import create_fake_dataframe
from app.group import group_rows
from app.nested.config import get_parallellism, get_target

from pyspark_tooling.spark import get_default_conf, get_sql_context
from pyspark_tooling.io import write_parquet
from pyspark_tooling.logger import log


def additional():

    "A minimal application designed to test deploying pyspark jobs to EMR"

    log.info("initiate additional step")

    parallellism = get_parallellism()

    spark = get_sql_context(get_default_conf("pyspark_tooling_test", parallellism))

    log.info("retrieved additional spark context successfullly", spark=spark)

    df = create_fake_dataframe(spark)

    log.info("created additional fake dataframe", record_count=df.count())

    # check our zipped code bundle is working
    res = group_rows(df)

    target = get_target(step="additional")

    log.info("writing additional dataframe to s3", target=target)

    write_parquet(target, res)


if __name__ == "__main__":
    additional()
