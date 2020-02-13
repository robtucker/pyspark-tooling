import pyspark.sql.functions as F
from nltk.metrics.distance import edit_distance as nltk_edit_distance
from pyspark.mllib.linalg import SparseVector, DenseVector
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from typing import Union


# Custom type hint for spark vectors
VectorType = Union[SparseVector, DenseVector]


def equality(primary_col: str, secondary_col: str, output_col: str, df: DataFrame):
    """A simple distance metric asserting that column a
    is equal to column b resulting in a double of 0.0 or 1.0"""
    return df.withColumn(
        output_col, (F.col(primary_col) == F.col(secondary_col)).cast(DoubleType())
    )


def quotient(primary_col: str, secondary_col: str, output_col: str, df: DataFrame):
    """The quotient is simply the minimum value divided by the maximum value
    Note that if the values are the same this will result in a score of 1.0,
    but if the values are very different this will result in scores close to 0.0"""

    return df.withColumn(
        output_col,
        F.when(
            F.col(primary_col).isNull() | F.col(secondary_col).isNull(), None
        ).otherwise(
            F.least(F.col(primary_col), F.col(secondary_col))
            / F.greatest(F.col(primary_col), F.col(secondary_col))
        ),
    )


def cosine_similarity(
    primary_col: str, secondary_col: str, output_col: str, df: DataFrame
):
    """Calculate the cosine similarity between 2 columns of vectors"""

    def _cosine(vector_a: VectorType, vector_b: VectorType):
        """Calculate the cosine similarity between 2 spark vectors"""
        if vector_a is None or vector_b is None:
            return None
        res = vector_a.dot(vector_b) / (vector_a.norm(2) * vector_b.norm(2))
        return res.item()

    func = F.udf(_cosine, returnType=DoubleType())

    return df.withColumn(output_col, func(F.col(primary_col), F.col(secondary_col)))


def jaccard_index(primary_col: str, secondary_col: str, output_col: str, df: DataFrame):
    """Calculate the intersection and union of two array columns"""

    return df.withColumn(
        output_col,
        F.when(
            F.col(primary_col).isNull() | F.col(secondary_col).isNull(), None
        ).otherwise(
            F.size(F.array_intersect(F.col(primary_col), F.col(secondary_col)))
            / F.size(F.array_union(F.col(primary_col), F.col(secondary_col)))
        ),
    )


def edit_distance(primary_col: str, secondary_col: str, output_col: str, df: DataFrame):
    """Calculate the edit distance between 2 columns of tokens"""

    def _edit(col_a: str, col_b: str):
        """Calculate the edit distance for 2 arrays of tokens"""
        if not col_a or not col_b:
            return None
        d = nltk_edit_distance(col_a, col_b)
        if d == 0:
            return 1.0  # strings are identical
        else:
            # return the distance divided by the max of the string lengths
            return 1 - (d / max(len(col_a), len(col_b)))

    func = F.udf(_edit, returnType=DoubleType())

    return df.withColumn(output_col, func(F.col(primary_col), F.col(secondary_col)))


def absolute_difference(
    primary_col: str, secondary_col: str, output_col: str, df: DataFrame
):
    """Return the absolute difference between 2 columns"""
    # note that sometimes the absolute function produces rounding errors
    return df.withColumn(
        output_col, F.round(F.abs(F.col(primary_col) - F.col(secondary_col)), 10)
    )
