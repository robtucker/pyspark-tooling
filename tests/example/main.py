from app.schema import create_fake_dataframe
from app.group import group_rows
from app.nested.config import get_parallellism, get_target

from pyspark_tooling.spark import get_default_conf, get_sql_context
from pyspark_tooling.io import write_parquet
from pyspark_tooling.logger import log


def main():

    "A minimal application designed to test deploying pyspark jobs to EMR"

    log.info("initiate application")

    parallellism = get_parallellism()

    spark = get_sql_context(get_default_conf("grada_pyspark_test", parallellism))

    log.info("retrieved spark context successfullly", spark=spark)

    df = create_fake_dataframe(spark)

    log.info("created fake dataframe", record_count=df.count())

    # check our zipped code bundle is working
    res = group_rows(df)

    target = get_target(step="main")

    log.info("writing dataframe to s3", target=target)

    write_parquet(target, res)


if __name__ == "__main__":
    main()
