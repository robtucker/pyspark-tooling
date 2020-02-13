import pyspark.sql.functions as F
from pyspark.sql import Column
from pyspark.sql.types import ArrayType, StringType


def length(col: Column):
    """Calculate the length of the column"""
    return F.size(col)


def remove_empty_strings(col: Column):
    """Remove empty strings and nulls from an array"""
    # note that nulls cannot be removed from arrays by using the
    # array_remove function because it relies on equality
    # and null != null, thus sadly we need a udf for this task
    remove_empties = F.udf(
        lambda x: [i for i in x if i and (i != "")] if x else [],
        returnType=ArrayType(StringType()),
    )
    return remove_empties(col)


def array_union(a: Column, b: Column) -> Column:
    """Calculate the union of two array columns"""
    return F.array_remove(F.array_union(a, b), "")


def array_intersection(a: Column, b: Column) -> Column:
    """Calculate the intersection of two array columns"""
    return F.array_remove(F.array_intersect(a, b), "")


def merge_collected_sets(a: Column, b: Column):
    """Merge 2 collected sets keeping only unique items"""

    def merge_col(list_a, list_b):
        set_a = set()
        set_b = set()
        if isinstance(list_a, list):
            set_a = set(list_a)

        if isinstance(list_b, list):
            set_b = set(list_b)

        return list(set_a.union(set_b))

    merger_udf = F.udf(merge_col, ArrayType(StringType()))

    return merger_udf(a, b)
