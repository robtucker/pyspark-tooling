import pytest
import numpy as np
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors

from app.nlp import vectors
from tests import base
from grada_pyspark_utils.dataframe import to_tuples


# @pytest.mark.focus
@pytest.mark.nlp
class TestVectors(base.BaseTest):
    @pytest.mark.usefixtures("spark")
    def test_term_frequency_vectors(self, spark: SQLContext):
        input_data = [
            (["foo", "foo", "bar"],),
            (["bar", "baz", "baz"],),
            (["baz", "foo"],),
        ]

        df = spark.createDataFrame(input_data, ["text"])
        res = vectors.term_frequency_vectors("text", "vectors", df, num_features=5)

        # with 5 buckets we get:
        # foo => 0
        # bar => 2
        # baz => 3

        expected_values = [
            a + b
            for a, b in zip(
                input_data,
                [
                    (Vectors.sparse(5, [0, 2], [2.0, 1.0]),),
                    (Vectors.sparse(5, [2, 3], [1.0, 2.0]),),
                    (Vectors.sparse(5, [0, 3], [1.0, 1.0]),),
                ],
            )
        ]

        self._validate_expected_values(res, ["text", "vectors"], expected_values)

    @pytest.mark.usefixtures("spark")
    def test_tfidf_vectors(self, spark: SQLContext):

        # based on the following imaginary term frequencies
        # [a, a, b]
        # [b, c, c]
        # [a, b, c, d]

        # these can be represented as sparse term frequency vectors
        input_data = [
            (Vectors.sparse(4, [0, 1], [2.0, 1.0]),),
            (Vectors.sparse(4, [1, 2], [1.0, 2.0]),),
            (Vectors.sparse(4, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0]),),
        ]

        # the idf calculation is:
        # ln((num docs + 1) / (num docs with term + 1))
        idf_a = np.log(4 / 3)
        idf_b = np.log(4 / 4)
        idf_c = np.log(4 / 3)
        idf_d = np.log(4 / 2)

        df = spark.createDataFrame(input_data, ["tf"])
        res = vectors.tfidf_vectors("tf", "tfidf", df)

        expected = [
            (Vectors.sparse(4, [0, 1], [2.0 * idf_a, 1.0 * idf_b]),),
            (Vectors.sparse(4, [1, 2], [1.0 * idf_b, 2.0 * idf_c]),),
            (
                Vectors.sparse(
                    4,
                    [0, 1, 2, 3],
                    [1.0 * idf_a, 1.0 * idf_b, 1.0 * idf_c, 1.0 * idf_d],
                ),
            ),
        ]

        self._validate_expected_values(res.select("tfidf"), ["tfidf"], expected)

    @pytest.mark.usefixtures("spark")
    def test_normalize_dense_vectors(self, spark: SQLContext):

        input_data = [(Vectors.dense([1, 4, 16]),), (Vectors.dense(1, 0, 9),)]
        df = spark.createDataFrame(input_data, ["vectors"])
        res = vectors.normalize_vectors("vectors", "normalized", df).select(
            "normalized"
        )

        vals = [i[0].toArray() for i in to_tuples(res)]

        # after being normalized the magnitude of each vector should be 1
        magnitudes = [np.linalg.norm(v) for v in vals]
        expected = [1.0 for _ in range(len(magnitudes))]
        # some magnitudes migth come out as 0.999999 etc
        self._validate_to_decimal_places(magnitudes, expected)

    @pytest.mark.usefixtures("spark")
    def test_normalize_sparse_vectors(self, spark: SQLContext):

        # based on the following imaginary tokens
        # [a, c, c]
        # [a, a, b]
        # [b, b, d]
        input_data = [
            (Vectors.sparse(4, [0, 2], [1.0, 2.0]),),
            (Vectors.sparse(4, [0, 1], [2.0, 1.0]),),
            (Vectors.sparse(4, [1, 3], [2.0, 1.0]),),
        ]

        df = spark.createDataFrame(input_data, ["vectors"])

        res = vectors.normalize_vectors("vectors", "normalized", df).select(
            "normalized"
        )

        vals = [i[0].toArray() for i in to_tuples(res)]

        # after being normalized the magnitude of each vector should be 1
        magnitudes = [np.linalg.norm(v) for v in vals]
        expected = [1.0 for _ in range(len(magnitudes))]
        # some magnitudes migth come out as 0.999999 etc
        self._validate_to_decimal_places(magnitudes, expected)

    @pytest.mark.usefixtures("spark")
    def test_sparse_vector_indices(self, spark: SQLContext):
        # based on the following tokens:
        # [a, a, c, d]
        # [b, c, c, c]
        # [a, b, d]
        # [c]
        input_data = [
            (0, Vectors.sparse(4, [0, 2, 3], [2.0, 1.0, 1.0])),
            (1, Vectors.sparse(4, [1, 2], [1.0, 3.0])),
            (2, Vectors.sparse(4, [0, 1, 3], [1.0, 3.0, 1.0])),
            (3, Vectors.sparse(4, [2], [1.0])),
        ]
        input_cols = ["id", "vectors"]
        df = spark.createDataFrame(input_data, input_cols)
        res = vectors.sparse_vector_indices("vectors", "indices", df)

        expected_values = [
            a + tuple([b])
            for a, b in zip(input_data, [[0, 2, 3], [1, 2], [0, 1, 3], [2]])
        ]

        expected_cols = input_cols + ["indices"]
        self._validate_expected_values(res, expected_cols, expected_values)
