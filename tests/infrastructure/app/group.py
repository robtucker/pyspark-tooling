from pyspark.sql import DataFrame
from app.schema import GROUP


def group_rows(df: DataFrame):
    return df.groupBy(F.col(GROUP)).agg()
