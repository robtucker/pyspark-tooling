import pytest
import numpy as np
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors as MllibVectors
from pyspark.ml.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, BlockMatrix

from app.nlp import matrix
from tests import base


# @pytest.mark.focus
@pytest.mark.nlp
class TestMatrix(base.BaseTest):
    """Test common matrix methods in spark"""

    @pytest.mark.usefixtures("spark")
    def test_df_to_indexed_row_matrix(self, spark: SQLContext):
        data = [
            (0, Vectors.dense(1.0, 2.0, 3.0)),
            (1, Vectors.dense(4.0, 5.0, 6.0)),
            (2, Vectors.dense(7.0, 8.0, 9.0)),
        ]
        cols = ["row_number", "vector"]

        df = spark.createDataFrame(data, cols)

        res = matrix.df_to_indexed_row_matrix("row_number", "vector", df)

        assert isinstance(res, IndexedRowMatrix)

        actual = [i.tolist() for i in res.toBlockMatrix().toLocalMatrix().toArray()]
        expected = [b.toArray().tolist() for a, b in data]

        assert actual == expected

    @pytest.mark.usefixtures("spark")
    def test_df_to_dense_matrix(self, spark: SQLContext):
        data = [
            (0, Vectors.dense(1.0, 2.0, 3.0)),
            (1, Vectors.dense(4.0, 5.0, 6.0)),
            (2, Vectors.dense(7.0, 8.0, 9.0)),
        ]
        cols = ["row_number", "vector"]

        df = spark.createDataFrame(data, cols)

        res = matrix.df_to_dense_matrix("row_number", "vector", df)

        actual = [i.tolist() for i in res.toArray()]
        expected = [b.toArray().tolist() for a, b in data]

        assert actual == expected

    @pytest.mark.usefixtures("spark")
    def test_df_to_block_matrix(self, spark: SQLContext):
        data = [
            (0, Vectors.dense(1.0, 2.0, 3.0)),
            (1, Vectors.dense(4.0, 5.0, 6.0)),
            (2, Vectors.dense(7.0, 8.0, 9.0)),
        ]
        cols = ["row_number", "vector"]

        df = spark.createDataFrame(data, cols)

        res = matrix.df_to_block_matrix("row_number", "vector", df)

        assert isinstance(res, BlockMatrix)

        actual = [i.tolist() for i in res.toLocalMatrix().toArray()]
        expected = [b.toArray().tolist() for a, b in data]

        assert actual == expected

    @pytest.mark.usefixtures("spark")
    def test_multiply_coordinate_matrices(self, spark: SQLContext):

        a_data = [(0, MllibVectors.dense(0, 3, 4)), (1, MllibVectors.dense(1, 2, 3))]

        b_data = [
            (0, MllibVectors.dense(1, 0)),
            (1, MllibVectors.dense(4, 2)),
            (2, MllibVectors.dense(1, 3)),
        ]

        matrix_a = IndexedRowMatrix(spark._sc.parallelize(a_data)).toCoordinateMatrix()

        matrix_b = IndexedRowMatrix(spark._sc.parallelize(b_data)).toCoordinateMatrix()

        product = matrix.multiply_coordinate_matrices(matrix_a, matrix_b)
        actual = product.toBlockMatrix().toLocalMatrix().toArray()

        expected = [[16.0, 18.0], [12.0, 13.0]]

        assert actual.tolist() == expected

    @pytest.mark.usefixtures("spark")
    def test_sparse_dot_product_cross_join(self, spark: SQLContext):
        # based on the following imaginary primary tokens
        # [a, a, b]
        # [b, c, c, c, d]
        primary_data = [
            (4, Vectors.sparse(4, [0, 1], [2.0, 1.0])),
            (5, Vectors.sparse(4, [1, 2, 3], [1.0, 3.0, 1.0])),
        ]
        # and the following imaginary secondary tokens
        # [a, c]
        # [a, b, c, d]
        secondary_data = [
            (7, Vectors.sparse(4, [0, 2], [1.0, 1.0])),
            (9, Vectors.sparse(4, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])),
        ]

        # if we were to write these as dense vectors:
        row_4 = np.array([2, 1, 0, 0])
        row_5 = np.array([0, 1, 3, 1])
        row_7 = np.array([1, 0, 1, 0])
        row_9 = np.array([1, 1, 1, 1])

        # calculate the expected dot product for each pair
        expected_values = [
            (4, 7, np.dot(row_4, row_7)),
            (4, 9, np.dot(row_4, row_9)),
            (5, 7, np.dot(row_5, row_7)),
            (5, 9, np.dot(row_5, row_9)),
        ]

        primary_cols = ["p_id", "p_vectors"]
        secondary_cols = ["s_id", "s_vectors"]
        primary_df = spark.createDataFrame(primary_data, primary_cols)

        secondary_df = spark.createDataFrame(secondary_data, secondary_cols)

        df = matrix.sparse_dot_product_cross_join(
            spark,
            "output",
            "p_id",
            "p_vectors",
            primary_df,
            "s_id",
            "s_vectors",
            secondary_df,
        )
        res = df.orderBy("p_id", "s_id")
        expected_cols = ["p_id", "s_id", "output"]
        self._validate_expected_values(res, expected_cols, expected_values)

    @pytest.mark.usefixtures("spark")
    def test_dense_dot_product_cross_join(self, spark: SQLContext):
        # based on the following imaginary primary tokens
        # [a, a, b]
        # [b, c, c, c, d]
        primary_data = [
            (4, Vectors.dense(2.0, 1.0, 0.0, 0.0)),
            (5, Vectors.dense(0.0, 1.0, 3.0, 1.0)),
        ]
        # and the following imaginary secondary tokens
        # [a, c]
        # [a, b, c, d]
        secondary_data = [
            (7, Vectors.dense(1.0, 0.0, 1.0, 0.0)),
            (9, Vectors.dense(1.0, 1.0, 1.0, 1.0)),
        ]

        # if we were to write these as dense vectors:
        row_4 = np.array([2, 1, 0, 0])
        row_5 = np.array([0, 1, 3, 1])
        row_7 = np.array([1, 0, 1, 0])
        row_9 = np.array([1, 1, 1, 1])

        # calculate the expected dot product for each pair
        expected_values = [
            (4, 7, np.dot(row_4, row_7)),
            (4, 9, np.dot(row_4, row_9)),
            (5, 7, np.dot(row_5, row_7)),
            (5, 9, np.dot(row_5, row_9)),
        ]

        primary_cols = ["p_id", "p_vectors"]
        secondary_cols = ["s_id", "s_vectors"]

        primary_df = spark.createDataFrame(primary_data, primary_cols)
        secondary_df = spark.createDataFrame(secondary_data, secondary_cols)

        df = matrix.dense_dot_product_cross_join(
            spark,
            "output",
            "p_id",
            "p_vectors",
            primary_df,
            "s_id",
            "s_vectors",
            secondary_df,
        )
        res = df.orderBy("p_id", "s_id")
        expected_cols = ["p_id", "s_id", "output"]
        self._validate_expected_values(res, expected_cols, expected_values)

    @pytest.mark.usefixtures("spark")
    def test_dense_matrix_cross_join(self, spark: SQLContext):
        # based on the following imaginary primary tokens
        # [a, a, b]
        # [b, c, c, c, d]
        primary_data = [
            (4, Vectors.dense(2.0, 1.0, 0.0, 0.0)),
            (5, Vectors.dense(0.0, 1.0, 3.0, 1.0)),
        ]
        # and the following imaginary secondary tokens
        # [a, c]
        # [a, b, c, d]
        secondary_data = [
            (7, Vectors.dense(1.0, 0.0, 1.0, 0.0)),
            (9, Vectors.dense(1.0, 1.0, 1.0, 1.0)),
        ]

        # if we were to write these as dense vectors:
        row_4 = np.array([2, 1, 0, 0])
        row_5 = np.array([0, 1, 3, 1])
        row_7 = np.array([1, 0, 1, 0])
        row_9 = np.array([1, 1, 1, 1])

        # calculate the expected dot product for each pair
        expected_values = [
            (4, 7, np.dot(row_4, row_7)),
            (4, 9, np.dot(row_4, row_9)),
            (5, 7, np.dot(row_5, row_7)),
            (5, 9, np.dot(row_5, row_9)),
        ]

        primary_cols = ["p_id", "p_vectors"]
        secondary_cols = ["s_id", "s_vectors"]

        primary_df = spark.createDataFrame(primary_data, primary_cols)
        secondary_df = spark.createDataFrame(secondary_data, secondary_cols)

        df = matrix.dense_matrix_cross_join(
            spark,
            "output",
            "p_id",
            matrix.df_to_block_matrix("p_id", "p_vectors", primary_df),
            "s_id",
            matrix.df_to_block_matrix("s_id", "s_vectors", secondary_df).transpose(),
        )

        res = df.orderBy("p_id", "s_id")

        expected_cols = ["p_id", "s_id", "output"]
        self._validate_expected_values(res, expected_cols, expected_values)

    @pytest.mark.usefixtures("spark")
    def test_jaccard_cross_join(self, spark: SQLContext):

        # based on the following imaginary primary tokens
        # [a, a, b]
        # [b, c, c]

        primary_data = [
            (0, Vectors.sparse(4, [0, 1], [2.0, 1.0])),
            (1, Vectors.sparse(4, [1, 2], [1.0, 2.0])),
        ]
        # and the following imaginary secondary tokens
        # [a, c]
        # [a, b, c, d]
        secondary_data = [
            (2, Vectors.sparse(4, [0, 2], [1.0, 1.0])),
            (3, Vectors.sparse(4, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])),
        ]

        cols = ["id", "vectors"]
        all_df = spark.createDataFrame(primary_data + secondary_data, cols)
        primary_df = spark.createDataFrame(primary_data, cols)
        secondary_df = spark.createDataFrame(secondary_data, cols)

        res = matrix.jaccard_cross_join(
            "vectors", "distances", all_df, primary_df, secondary_df
        )
        res.show()

        # TODO - get this min hash version of jaccard working!!
        # calculate the jaccard index as the
        # intersection divided by the union
        # expected_data = [
        #     # [a, a, b] vs [a, c]
        #     (0, 2, 1 / 3),
        #     # [a, a, b] vs [a, b, c, d]
        #     (0, 3, 2 / 4),
        #     # [b, c, c] vs [a, c]
        #     (1, 2, 1 / 3),
        #     # [b, c, c] vs [a, b, c, d]
        #     (1, 3, 2 / 4),
        # ]
