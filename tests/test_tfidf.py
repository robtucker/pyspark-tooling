import pytest
from pyspark.sql import SQLContext
from pyspark.ml.linalg import SparseVector

from app.nlp import tokens, pipelines
from tests import base
from grada_pyspark_utils.dataframe import to_tuples


# @pytest.mark.focus
@pytest.mark.nlp
class TestPipelines(base.BaseTest):
    """Test pipelines for creating NLP vectors"""

    @pytest.mark.usefixtures("spark")
    def test_token_vectors_pipeline(self, spark: SQLContext):
        input_data = [
            ("foo bar baz biz",),
            ("foo    baz   bar",),
            ("bar baz  ",),
            ("  foo  biz  ",),
            ("",),
            (None,),
        ]

        raw = spark.createDataFrame(input_data, ["text"])
        res = pipelines.token_vectors_pipeline("text", "vectors", raw)

        actual = to_tuples(res.select("vectors"))

        row_0 = set(actual[0][0])
        row_1 = set(actual[1][0])
        row_2 = set(actual[2][0])
        row_3 = set(actual[3][0])
        row_4 = set(actual[4][0])
        row_5 = set(actual[5][0])

        assert len(row_0) == 4
        assert len(row_1) == 3
        assert len(row_2) == 2
        assert len(row_3) == 2
        assert len(row_4) == 0
        assert len(row_5) == 0

        assert row_1.issubset(row_0)
        assert row_2.issubset(row_0)
        assert row_3.issubset(row_0)

        assert len(row_1.intersection(row_2)) == 2
        assert len(row_2.intersection(row_3)) == 0

    @pytest.mark.usefixtures("spark")
    def test_stemmed_token_vectors_pipeline(self, spark: SQLContext):
        # the exact same set up as the previous test however we are expecting
        # the stemmer to reduce these 4 words to: run, walk, jog, sprint
        input_data = [
            ("running walks jogged sprinted",),
            ("runs jogging walked",),
            ("walking jogs",),
            ("running sprinting",),
            ("",),
            (None,),
        ]

        raw = spark.createDataFrame(input_data, ["text"])

        res = pipelines.token_vectors_pipeline(
            "text", "vectors", raw, stemmer_func=tokens.porter_tokens
        )

        actual = to_tuples(res.select("vectors"))

        row_0 = set(actual[0][0])
        row_1 = set(actual[1][0])
        row_2 = set(actual[2][0])
        row_3 = set(actual[3][0])
        row_4 = set(actual[4][0])
        row_5 = set(actual[5][0])

        assert len(row_0) == 4
        assert len(row_1) == 3
        assert len(row_2) == 2
        assert len(row_3) == 2
        assert len(row_4) == 0
        assert len(row_5) == 0

        assert row_1.issubset(row_0)
        assert row_2.issubset(row_0)
        assert row_3.issubset(row_0)

        assert len(row_1.intersection(row_2)) == 2
        assert len(row_1.intersection(row_3)) == 1
        assert len(row_2.intersection(row_3)) == 0

    @pytest.mark.usefixtures("spark")
    def test_tf_ngrams_pipeline(self, spark: SQLContext):
        input_data = [
            ("foo bar baz biz",),
            ("foo baz bar",),
            ("bar baz",),
            ("foo biz",),
            ("",),
            (None,),
        ]
        raw = spark.createDataFrame(input_data, ["text"])
        res = pipelines.tf_ngrams_pipeline("text", "vectors", raw)

        actual = [i[0] for i in to_tuples(res.select("vectors"))]

        for v in actual:
            assert isinstance(v, SparseVector)

        row_0 = set(actual[0].indices)
        row_1 = set(actual[1].indices)
        row_2 = set(actual[2].indices)
        row_3 = set(actual[3].indices)
        row_4 = set(actual[4].indices)
        row_5 = set(actual[4].indices)

        assert row_1.issubset(row_0)
        assert row_2.issubset(row_0)
        assert row_3.issubset(row_0)

        assert len(row_4) == 0
        assert len(row_5) == 0

    @pytest.mark.usefixtures("spark")
    def test_tfidf_vectors_pipeline(self, spark: SQLContext):
        input_data = [
            ("foo bar baz biz",),
            ("foo baz bar",),
            ("bar baz",),
            ("foo biz",),
            ("",),
            (None,),
        ]
        raw = spark.createDataFrame(input_data, ["text"])
        res = pipelines.tfidf_vectors_pipeline("text", "vectors", raw)

        actual = [i[0] for i in to_tuples(res.select("vectors"))]

        for v in actual:
            assert isinstance(v, SparseVector)

        row_0 = set(actual[0].indices)
        row_1 = set(actual[1].indices)
        row_2 = set(actual[2].indices)
        row_3 = set(actual[3].indices)
        row_4 = set(actual[4].indices)
        row_5 = set(actual[4].indices)

        assert row_1.issubset(row_0)
        assert row_2.issubset(row_0)
        assert row_3.issubset(row_0)

        assert len(row_4) == 0
        assert len(row_5) == 0
