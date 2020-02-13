import pytest
from pyspark.sql import SQLContext
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from app.nlp import tokens
from app.schema import schema
from grada_pyspark_utils.dataframe import to_tuples
from tests import base


# @pytest.mark.focus
@pytest.mark.nlp
class TestTokens(base.BaseTest):
    """Test tokenization and stemming of input strings"""

    @pytest.mark.usefixtures("spark")
    def test_tokenize_words(self, spark: SQLContext):
        input_data = [("Chocolate Cake",), ("Don't panic!",), ("",), ("",)]

        raw = spark.createDataFrame(input_data, ["text"])

        df = tokens.tokenize_words("text", "tokens", raw)
        res = df.select("tokens")

        # note that the tokenizer will not remove empty strings
        # the remove empty strings from tokens function should be used
        expected_values = [
            (["chocolate", "cake"],),
            (["don't", "panic!"],),
            ([""],),
            ([""],),
        ]

        self._validate_expected_values(res, ["tokens"], expected_values)

    @pytest.mark.usefixtures("spark")
    def test_string_to_character_array(self, spark: SQLContext):
        data = [("fuzzy wuzzy was a bear",)]

        df = spark.createDataFrame(data, ["text"])

        df_tokenised = tokens.string_to_character_array(
            "text", "tokenised_characters", df
        )

        expected_values = [(i[0], [j for j in i[0]]) for i in data]

        expected_cols = ["text", "tokenised_characters"]

        self._validate_expected_values(df_tokenised, expected_cols, expected_values)

    @pytest.mark.usefixtures("spark")
    def test_ngrams(self, spark: SQLContext):
        input_data = [(["to", "infinity", "and", "beyond"],)]

        raw = spark.createDataFrame(input_data, ["tokens"])

        df = tokens.ngrams("tokens", "ngrams", raw, n=2)

        expected_values = [
            a + b
            for a, b in zip(
                input_data, [(["to infinity", "infinity and", "and beyond"],)]
            )
        ]

        expected_cols = ["tokens", "ngrams"]
        self._validate_expected_values(df, expected_cols, expected_values)

    @pytest.mark.usefixtures("spark")
    def test_character_ngrams(self, spark):
        n = 3
        data = [("Sainsbury's Chocolate Cake",), ("Don't panic!",), ("",), (None,)]

        df = spark.createDataFrame(data, ["text"])
        res = tokens.character_ngrams("text", "trigrams", df, n=n)

        expected = [
            (i[0], [i[0][j : j + n] for j in range(0, len(i[0], n))] if i[0] else None)
            for i in data
        ]

        assert actual == expected

    @pytest.mark.usefixtures("spark")
    def test_sort_tokens(self, spark: SQLContext):
        data = [(["world", "hello"],), (["the", "cat", "sat", "on", "the", "mat"],)]

        in_col = "tokens"
        out_col = "res"
        raw = spark.createDataFrame(data, [in_col])
        df = tokens.sorted_tokens(in_col, out_col, raw)

        expected_values = [
            (["world", "hello"], ["hello", "world"]),
            (
                ["the", "cat", "sat", "on", "the", "mat"],
                ["cat", "mat", "on", "sat", "the", "the"],
            ),
        ]

        self._validate_expected_values(df, [in_col, out_col], expected_values)

    @pytest.mark.usefixtures("spark")
    def test_porter_stemmer(self, spark: SQLContext):
        raw = self._get_stemmer_input(spark)
        output_col = "res"
        df = tokens.porter_tokens("tokens", output_col, raw)

        expected = [
            (["I", "may", "be", "use"],),
            (["a", "simplist", "stem", "algorithm"],),
            (["but", "the", "result", "are", "great"],),
        ]

        assert to_tuples(df.select(output_col)) == expected

    @pytest.mark.usefixtures("spark")
    def test_lancaster_stemmer(self, spark: SQLContext):
        raw = self._get_stemmer_input(spark)
        output_col = "res"
        df = tokens.lancaster_tokens("tokens", output_col, raw)

        expected = [
            (["i", "may", "be", "us"],),
            (["a", "simpl", "stem", "algorithm"],),
            (["but", "the", "result", "ar", "gre"],),
        ]

        assert to_tuples(df.select(output_col)) == expected

    @pytest.mark.usefixtures("spark")
    def test_snowball_stemmer(self, spark: SQLContext):
        raw = self._get_stemmer_input(spark)
        output_col = "res"
        df = tokens.snowball_tokens(schema.PRODUCT_NAME_TOKENS, output_col, raw)

        expected = [
            (["i", "may", "be", "use"],),
            (["a", "simplist", "stem", "algorithm"],),
            (["but", "the", "result", "are", "great"],),
        ]

        assert to_tuples(df.select(output_col)) == expected

    @pytest.mark.usefixtures("spark")
    def test_fill_nulls_with_empty_string(self, spark: SQLContext):
        input_data = [("a",), (None,)]
        df = spark.createDataFrame(input_data, ["text"])

        res = tokens.fill_nulls_with_empty_string("text", "filled", df)

        expected_values = [a + b for a, b in zip(input_data, [("a",), ("",)])]

        self._validate_expected_values(res, ["text", "filled"], expected_values)

    @pytest.mark.usefixtures("spark")
    def test_rm_empty_strings_from_tokens(self, spark: SQLContext):
        input_data = [(["a", "", None],), (["", None],), (None,)]
        df = spark.createDataFrame(input_data, ["text"])

        res = tokens.rm_empty_strings_from_tokens("text", "cleaned", df)

        expected_values = [a + b for a, b in zip(input_data, [(["a"],), ([],), ([],)])]

        self._validate_expected_values(res, ["text", "cleaned"], expected_values)

    def _get_stemmer_input(self, spark: SQLContext):
        data = [
            (["I", "may", "be", "using"],),
            (["a", "simplistic", "stemming", "algorithm"],),
            (["but", "the", "results", "are", "great"],),
        ]
        s = StructType(
            [StructField(schema.PRODUCT_NAME_TOKENS, ArrayType(StringType()))]
        )

        return spark.createDataFrame(data, s)
