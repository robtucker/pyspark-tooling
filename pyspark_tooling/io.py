from pyspark import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql.types import DataType

from grada_pyspark_utils.validators import validate_schema


def read_csv(spark: SparkContext, path: str, schema=None):
    """Load csv files from the source directory into a dataframe"""
    df = spark.read.csv(path, header=True, schema=schema)

    if not schema:
        return df

    # recreate the dataframe with the correct schema
    # this will ensure non-nullable columns do not contain null values
    return spark.createDataFrame(df.rdd, schema, verifySchema=True)


def write_csv(
    path: str,
    df: DataFrame,
    schema: DataType = None,
    coalesce: int = None,
    partition_by=None,
    max_records_per_file=None,
    mode="overwrite",
):
    """Write a dataframe to csv format"""

    writer = _get_writer(
        df,
        schema=schema,
        coalesce=coalesce,
        partition_by=partition_by,
        max_records_per_file=max_records_per_file,
    )

    writer.save(path, format="csv", header=True, mode=mode, quote="\u0000")


def read_parquet(
    spark: SparkContext, path: str, schema: DataType = None, merge_schema=True
):
    """Read a directory of parquet files into a dataframe"""

    # initially all the columns will be nullable
    df = spark.read.option("mergeSchema", str(merge_schema).lower()).parquet(path)

    if not schema:
        # return the dataframe without validating the schema
        return df

    # recreate the dataframe with the correct schema
    # this will ensure non-nullable columns do not contain null values
    return spark.createDataFrame(df.rdd, schema, verifySchema=True)


def write_parquet(
    path: str,
    df: DataFrame,
    schema: DataType = None,
    coalesce: int = None,
    partition_by=None,
    max_records_per_file=None,
    mode="overwrite",
):
    """Write a dataframe to parquet format"""
    writer = _get_writer(
        df,
        schema=schema,
        coalesce=coalesce,
        partition_by=partition_by,
        max_records_per_file=max_records_per_file,
    )

    writer.parquet(path, mode=mode)


def write_jdbc(
    url: str,
    table: str,
    user: dict,
    password: str,
    df: DataFrame,
    schema: DataType = None,
    mode="error",
):
    """Write a dataframe to jdbc"""
    if schema:
        validate_schema(df, schema)

    df.write.format("jdbc").options(
        url=url,
        dbtable=table,
        user=user,
        password=password,
        driver="org.postgresql.Driver",
        mode=mode,
    ).save()


def _get_writer(
    df: DataFrame,
    schema: DataType = None,
    coalesce: int = None,
    partition_by=None,
    max_records_per_file=None,
):
    if schema:
        validate_schema(df, schema)

    if coalesce:
        df = df.coalesce(coalesce)

    w = df.write

    if partition_by:
        w = w.partitionBy(partition_by)

    if max_records_per_file:
        w = w.option("maxRecordsPerFile", max_records_per_file)

    return w
