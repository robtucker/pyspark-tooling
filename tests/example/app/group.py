from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from app.schema import GROUP, GROUP_COUNT


def group_rows(df: DataFrame):
    """aggregate each group and count the rows"""
    return df.groupBy(F.col(GROUP)).agg(F.count(F.lit(1)).alias(GROUP_COUNT))
