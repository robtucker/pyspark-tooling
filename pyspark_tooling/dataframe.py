import pytz
from datetime import datetime

from pyspark.sql import DataFrame, Column, Row, Window
import pyspark.sql.functions as F
from grada_pyspark_utils.types import ColumnList


def to_lists(df: DataFrame) -> list:
    """Convert a dataframe to a list of lists
    TEST PURPOSES ONLY - this method will collect the dataframe in the driver"""
    return [
        _parse_row(row, row_as_tuple=False, array_as_tuple=False)
        for row in df.collect()
    ]


def to_tuples(df: DataFrame, array_as_tuple=False) -> list:
    """Convert a dataframe to a list of tuples
    TEST PURPOSES ONLY - this method will collect the dataframe in the driver"""
    return [
        _parse_row(row, row_as_tuple=True, array_as_tuple=array_as_tuple)
        for row in df.collect()
    ]


def to_dicts(df: DataFrame) -> list:
    """Convert a dataframe to an list of dictionaries
    TEST PURPOSES ONLY - this method will collect the dataframe in the driver"""
    return df.rdd.map(lambda row: row.asDict()).collect()


def lowercase_column_names(df: DataFrame) -> DataFrame:
    """Convert the column names of a dataframe to lowercase"""
    return df.toDF(*[c.lower() for c in df.columns])


def a_not_in_b(join_fields: list, a: DataFrame, b: DataFrame):
    """Return all rows in A that are not found in B"""
    res = a.join(b, join_fields, "left_anti")
    # return the df with the original column order
    return res.select(a.columns)


def with_row_number(
    output_col: str, order_by: list, df: DataFrame, sort="asc", zero_indexed=True
) -> DataFrame:
    """Assign a sequential row number to each member of a dataframe"""

    is_desc = sort.lower() in ["desc", "descending"]
    if isinstance(order_by, str) or isinstance(order_by, Column):
        order_by = [order_by]
    elif not isinstance(order_by, list):
        msg = "Ordering criteria must be a string column name or a list of string column names"
        raise Exception(msg)

    # create a window function depending on the sort order
    if is_desc:
        window = Window.orderBy(*[F.desc(i) for i in order_by])
    else:
        window = Window.orderBy(*[F.asc(i) for i in order_by])

    # if the client wants to start from row 1 then that's fine
    if not zero_indexed:
        return df.withColumn(output_col, F.row_number().over(window))

    # otherwise start from row number 0
    return df.withColumn(output_col, F.row_number().over(window) - 1)


def make_col_nullable(col_name: str, df: DataFrame):
    return df.withColumn(
        col_name,
        F.when(F.col(col_name).isNotNull(), F.col(col_name)).otherwise(F.lit(None)),
    )


def select_cols(columns: ColumnList, df: DataFrame) -> DataFrame:
    """Select a list of Column class objects or column names"""
    return df.select(*[F.col(i) if isinstance(i, str) else i for i in columns])


def drop_cols(columns: ColumnList, df: DataFrame) -> DataFrame:
    """Drop a list of Column class objects or column names"""
    return df.drop(*columns)


def union(a: DataFrame, b: DataFrame) -> DataFrame:
    """Union 2 dataframes"""
    return a.union(b)


def repartition(parallellism: int, df: DataFrame) -> DataFrame:
    """Repartition using the configure number of partitions"""
    return df.repartition(parallellism)


def deduplicate(df: DataFrame) -> DataFrame:
    """Remove duplicates from the dataframe"""
    return df.dropDuplicates()


def join(a: DataFrame, b: DataFrame, on: list, how="left") -> DataFrame:
    """Return the cartesian product of a and b"""
    return a.join(b, on=on, how=how)


def _parse_row(row, row_as_tuple=True, array_as_tuple=False):
    """Recursively convert a pyspark row into a list of tuples"""
    # array types should always be returned as lists
    if isinstance(row, list):
        res = [
            _parse_row(i, row_as_tuple=row_as_tuple, array_as_tuple=array_as_tuple)
            for i in row
        ]
        if array_as_tuple:
            return tuple(res)
        return res
    # rows will be converted to tuples by default
    if isinstance(row, Row):
        res = [
            _parse_row(i, row_as_tuple=row_as_tuple, array_as_tuple=array_as_tuple)
            for i in row
        ]
        if row_as_tuple:
            return tuple(res)
        return res

    if isinstance(row, datetime):
        # datetimes come back as naive datetimes in local time
        return row.astimezone(pytz.utc)

    # other literal values should be returned verbatim
    return row
