import pytest
from pyspark.sql import SQLContext
from pyspark.ml.linalg import Vectors

from app.nlp import distance
from app.config import Config
from app.features.spacy import Spacy
from grada_pyspark_utils.dataframe import to_dicts, to_tuples
from tests import base


# @pytest.mark.focus
@pytest.mark.nlp
class TestDistance(base.BaseTest):
    """Test various distance metrics used in scoring"""

    @pytest.mark.usefixtures("spark")
    def test_quotient(self, spark: SQLContext):
        primary_col = "col_a"
        secondary_col = "col_b"
        input_cols = [primary_col, secondary_col]
        output_col = "res"

        input_data = [(1.0, 10.0), (5.5, 27.5), (27.6, 92.0), (14.75, 5.9)]

        df = spark.createDataFrame(input_data, input_cols)
        res = distance.quotient(primary_col, secondary_col, output_col, df)

        distances = [0.1, 0.2, 0.3, 0.4]

        expected_values = [a + (b,) for a, b in zip(input_data, distances)]
        expected_cols = input_cols + [output_col]

        self._validate_expected_values(res, expected_cols, expected_values)

    @pytest.mark.usefixtures("spark")
    def test_cosine_similarity(self, spark: SQLContext):

        input_data = [
            # two_dim_normals
            (Vectors.dense([1, 0]), Vectors.dense([0, 1])),
            # three_dim_normals
            (Vectors.dense([0, 1, 0]), Vectors.dense([0, 0, 1])),
            # two_dim_colinear
            (Vectors.dense([1, 0]), Vectors.dense([1, 0])),
            # three_dim_colinear
            (Vectors.dense([1, 1, 0]), Vectors.dense([1, 1, 0])),
        ]

        df = spark.createDataFrame(input_data, ["col_a", "col_b"])
        res = distance.cosine_similarity("col_a", "col_b", "col_c", df)

        actual = [i[0] for i in to_tuples(res.select("col_c"))]
        expected = [0.0, 0.0, 1.0, 1.0]
        self._validate_to_decimal_places(actual, expected, decimal_places=6)

    @pytest.mark.usefixtures("spark", "conf")
    def test_spacy_cosine_similarity(
        self, spark: SQLContext, conf: Config, spacy: Spacy
    ):
        """Confirm that the pyspark cosine calculations are
        the same as the spacy cosine calculations"""

        primary_col = "primary_vectors"
        secondary_col = "secondary_vectors"
        output_col = "res"

        nlp = spacy.get_spacy()

        texts = ["ale", "rum", "mojito", "beer", "lager", "vodka"]
        docs = list(nlp.pipe(texts))

        vectors = [Vectors.dense(doc.vector.tolist()) for doc in docs]
        input_data = []
        for i in range(len(vectors)):
            input_data.append((vectors[0], vectors[i]))

        df = spark.createDataFrame(input_data, [primary_col, secondary_col])

        res = distance.cosine_similarity(primary_col, secondary_col, output_col, df)

        actual = [i[output_col] for i in to_dicts(res)]
        expected = [docs[0].similarity(doc) for doc in docs]

        # soacy and pyspark must give the same value to at least 6 decimal places
        self._validate_to_decimal_places(actual, expected, decimal_places=6)

    @pytest.mark.usefixtures("spark")
    def test_strings_jaccard_index(self, spark: SQLContext):
        primary_col = "col_a"
        secondary_col = "col_b"
        input_cols = [primary_col, secondary_col]
        output_col = "res"

        input_data = [
            (["a", "b"], ["b", "c", "d"]),
            (["x", "y"], ["x", "y", "z"]),
            (["a", "b", "d"], ["a", "b", "c", "d"]),
            (["a", "b"], []),
            (["a", "b"], None),
        ]

        df = spark.createDataFrame(input_data, input_cols)
        res = distance.jaccard_index(primary_col, secondary_col, output_col, df)

        distances = [1 / 4, 2 / 3, 3 / 4, 0.0, None]
        expected_values = [(a + (b,)) for a, b in zip(input_data, distances)]
        expected_cols = input_cols + [output_col]

        self._validate_expected_values(res, expected_cols, expected_values)

    @pytest.mark.usefixtures("spark")
    def test_integers_jaccard_index(self, spark: SQLContext):
        primary_col = "col_a"
        secondary_col = "col_b"
        input_cols = [primary_col, secondary_col]
        output_col = "res"

        input_data = [
            ([1, 4], [2, 3, 4]),
            ([1, 2], [1, 2, 3]),
            ([1, 2, 4], [1, 2, 3, 4]),
            ([1, 2], []),
            ([1, 2], None),
        ]

        df = spark.createDataFrame(input_data, input_cols)
        res = distance.jaccard_index(primary_col, secondary_col, output_col, df)

        distances = [1 / 4, 2 / 3, 3 / 4, 0.0, None]
        expected_values = [(a + (b,)) for a, b in zip(input_data, distances)]
        expected_cols = input_cols + [output_col]

        self._validate_expected_values(res, expected_cols, expected_values)

    @pytest.mark.usefixtures("spark")
    def test_edit_distance(self, spark: SQLContext):
        primary_col = "col_a"
        secondary_col = "col_b"
        input_cols = [primary_col, secondary_col]
        output_col = "res"

        input_data = [
            ("sandwich", "sandwich"),
            ("apple cider", "apple pie"),
            ("lion bar", "kitkat"),
            ("sparkling wine", ""),
            (None, "sparkling water"),
        ]

        df = spark.createDataFrame(input_data, input_cols)
        res = distance.edit_distance(primary_col, secondary_col, output_col, df)

        distances = [1.0, 1 - (3 / 11), 1 - (6 / 8), None, None]
        expected_values = [a + (b,) for a, b in zip(input_data, distances)]
        expected_cols = input_cols + [output_col]

        self._validate_expected_values(res, expected_cols, expected_values)
