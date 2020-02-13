import pytest

import pyspark.sql.functions as F
from pyspark.sql.types import Row, StructType, StructField, StringType

from grada_pyspark_utils import dataframe

from tests import base


# @pytest.mark.focus
class TestDataframeUtils(base.BaseTest):
    """Test dataframe helpers"""

    @pytest.mark.usefixtures("spark")
    def test_parse_row(self, spark):
        rows = [Row("a", "b", "c"), Row("d", "e", "f"), Row("g", "h", "i")]
        expectedList = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
        expectedTuple = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")]

        assert dataframe._parse_row(rows, row_as_tuple=False) == expectedList
        assert dataframe._parse_row(rows, row_as_tuple=True) == expectedTuple

    @pytest.mark.usefixtures("spark")
    def test_to_lists(self, spark):
        cols = ["col_a", "col_b", "col_c"]
        data = [("a", 2, "c"), ("d", 5, "f"), ("g", 8, "i")]
        df = spark.createDataFrame(data, cols)
        expected = [["a", 2, "c"], ["d", 5, "f"], ["g", 8, "i"]]

        assert dataframe.to_lists(df) == expected

    @pytest.mark.usefixtures("spark")
    def test_to_tuples(self, spark):
        cols = ["col_a", "col_b", "col_c"]
        data = [("a", 2, "c"), ("d", 5, "f"), ("g", 8, "i")]
        df = spark.createDataFrame(data, cols)
        expected = [("a", 2, "c"), ("d", 5, "f"), ("g", 8, "i")]

        assert dataframe.to_tuples(df) == expected

    @pytest.mark.usefixtures("spark")
    def test_to_dicts(self, spark):
        cols = ["col_a", "col_b", "col_c"]
        data = [("a", 2, "c"), ("d", 5, "f"), ("g", 8, "i")]
        df = spark.createDataFrame(data, cols)
        expected = [
            {"col_a": "a", "col_b": 2, "col_c": "c"},
            {"col_a": "d", "col_b": 5, "col_c": "f"},
            {"col_a": "g", "col_b": 8, "col_c": "i"},
        ]

        assert dataframe.to_dicts(df) == expected

    @pytest.mark.usefixtures("spark")
    def test_row_number_by_single_col(self, spark):
        input_col = "input_col"
        output_col = "output_col"
        expected_cols = [input_col, output_col]
        input_data = [tuple(i) for i in ("a", "b", "c", "d", "e")]

        raw = spark.createDataFrame(input_data, [input_col])

        ascending = dataframe.with_row_number(output_col, input_col, raw)
        descending = dataframe.with_row_number(output_col, input_col, raw, sort="desc")

        expected_ascending = [("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)]
        expected_descending = [("e", 0), ("d", 1), ("c", 2), ("b", 3), ("a", 4)]

        self.validate_values(ascending, expected_cols, expected_ascending)
        self.validate_values(descending, expected_cols, expected_descending)

    @pytest.mark.usefixtures("spark")
    def test_row_number_by_multiple_cols(self, spark):
        input_cols = ["col_a", "col_b"]
        output_col = "output_col"
        expected_cols = input_cols + [output_col]
        input_data = [
            ("a", "x"),
            ("a", "y"),
            ("a", "z"),
            ("b", "x"),
            ("b", "y"),
            ("b", "z"),
        ]

        raw = spark.createDataFrame(input_data, input_cols)

        ascending = dataframe.with_row_number(output_col, input_cols, raw)
        descending = dataframe.with_row_number(output_col, input_cols, raw, sort="desc")

        expected_ascending = [
            ("a", "x", 0),
            ("a", "y", 1),
            ("a", "z", 2),
            ("b", "x", 3),
            ("b", "y", 4),
            ("b", "z", 5),
        ]
        expected_descending = [
            ("b", "z", 0),
            ("b", "y", 1),
            ("b", "x", 2),
            ("a", "z", 3),
            ("a", "y", 4),
            ("a", "x", 5),
        ]

        self.validate_values(ascending, expected_cols, expected_ascending)
        self.validate_values(descending, expected_cols, expected_descending)

    @pytest.mark.usefixtures("spark")
    def test_union(self, spark):
        df_a_cols = ["col_1", "col_2", "col_3"]
        df_a_data = [("a", "b", "c")]
        df_a = spark.createDataFrame(df_a_data, df_a_cols)
        df_b_cols = ["col_1", "col_2", "col_3"]
        df_b_data = [("d", "e", "f")]
        df_b = spark.createDataFrame(df_b_data, df_b_cols)

        expected_cols = ["col_1", "col_2", "col_3"]
        expected_data = [("a", "b", "c"), ("d", "e", "f")]

        self.validate_values(dataframe.union(df_a, df_b), expected_cols, expected_data)

    @pytest.mark.usefixtures("spark")
    def test_repartition(self, spark):
        cols = ["col_a", "col_b", "col_c"]
        data = [
            ("a", "b", 3),
            ("d", "e", 6),
            ("g", "h", 9),
            ("j", "k", 12),
            ("m", "n", 15),
            ("p", "q", 18),
        ]
        df = spark.createDataFrame(data, cols)
        df = dataframe.repartition(4, df)
        assert df.rdd.getNumPartitions() == 4

    @pytest.mark.usefixtures("spark")
    def test_deduplicate(self, spark):
        cols = ["col_a", "col_b", "col_c"]
        data = [
            ("a", "b", 3),
            ("d", "e", 6),
            ("a", "b", 3),
            ("j", "k", 12),
            ("m", "b", 15),
            ("d", "e", 6),
            ("p", "q", 18),
            ("p", "q", 18),
            ("s", "t", 3),
        ]
        df = spark.createDataFrame(data, cols).orderBy(cols)

        expected_cols = ["col_a", "col_b", "col_c"]
        expected_data = [
            ("a", "b", 3),
            ("d", "e", 6),
            ("j", "k", 12),
            ("m", "b", 15),
            ("p", "q", 18),
            ("s", "t", 3),
        ]

        self.validate_values(dataframe.deduplicate(df), expected_cols, expected_data)

    @pytest.mark.usefixtures("spark")
    def test_make_col_nullable(self, spark):
        col = "text"
        data = [("a",), ("b",), ("c",)]
        input_schema = StructType([StructField(col, StringType(), False)])

        ouput_schema = StructType([StructField(col, StringType(), True)])

        df = spark.createDataFrame(data, input_schema)

        res = dataframe.make_col_nullable(col, df)

        self.validate_schema(res, ouput_schema)

    @pytest.mark.usefixtures("spark")
    def test_lowercase_column_names(self, spark):
        s = StructType(
            [StructField("AbC", StringType()), StructField("dEF", StringType())]
        )

        df = spark.createDataFrame([("foo", "bar")], s)

        res = dataframe.lowercase_column_names(df)

        assert res.columns == ["abc", "def"]

    @pytest.mark.usefixtures("spark")
    def test_a_not_in_b(self, spark):
        cols_a_and_b = ["col_a", "col_b", "col_c"]
        data_a = [
            ("a", "b", 3),
            ("d", "e", 6),
            ("g", "h", 9),
            ("j", "k", 12),
            ("m", "n", 15),
            ("p", "q", 18),
        ]
        data_b = [
            ("a", "b", 3),
            ("d", "h", 6),
            ("g", "h", 12),
            ("j", "k", 12),
            ("a", "n", 15),
            ("p", "r", 18),
        ]
        join_cols = ["col_a", "col_b"]
        df_a = spark.createDataFrame(data_a, cols_a_and_b)
        df_b = spark.createDataFrame(data_b, cols_a_and_b)

        expected_df_cols = ["col_a", "col_b", "col_c"]
        expected_df_data = [("d", "e", 6), ("m", "n", 15), ("p", "q", 18)]

        actual_df = dataframe.a_not_in_b(join_cols, df_a, df_b)

        self.validate_values(
            actual_df.orderBy(["col_a"]), expected_df_cols, expected_df_data
        )

    @pytest.mark.usefixtures("spark")
    def test_join_segments(self, spark):
        a_cols = ["key_a", "category_a", "value_a"]
        a_data = [
            ("a", "category_1", 1),
            ("b", "category_1", 2),
            ("c", "category_2", 3),
            ("d", "category_2", 4),
        ]

        b_cols = ["key_b", "category_b", "value_b"]
        b_data = [
            ("w", "category_1", 6),
            ("x", "category_1", 7),
            ("y", "category_2", 8),
            ("z", "category_2", 9),
        ]

        a_df = spark.createDataFrame(a_data, a_cols)
        b_df = spark.createDataFrame(b_data, b_cols)

        conditions = [F.col("category_a") == F.col("category_b")]
        df = dataframe.join(a_df, b_df, conditions).orderBy(["key_a", "key_b"])

        expected_cols = [
            "key_a",
            "category_a",
            "value_a",
            "key_b",
            "category_b",
            "value_b",
        ]
        expected_data = [
            ("a", "category_1", 1, "w", "category_1", 6),
            ("a", "category_1", 1, "x", "category_1", 7),
            ("b", "category_1", 2, "w", "category_1", 6),
            ("b", "category_1", 2, "x", "category_1", 7),
            ("c", "category_2", 3, "y", "category_2", 8),
            ("c", "category_2", 3, "z", "category_2", 9),
            ("d", "category_2", 4, "y", "category_2", 8),
            ("d", "category_2", 4, "z", "category_2", 9),
        ]

        self.validate_values(df, expected_cols, expected_data)

    @pytest.mark.usefixtures("spark")
    def test_select_cols(self, spark):
        cols = ["col_a", "col_b,", "col_c", "col_d", "col_e"]
        data = [("a", "b", "c", "d", "e")]
        df = spark.createDataFrame(data, cols)
        selected_cols = dataframe.select_cols(["col_a", "col_c", "col_e"], df)
        expected_cols = ["col_a", "col_c", "col_e"]
        expected_data = [("a", "c", "e")]

        self.validate_values(selected_cols, expected_cols, expected_data)

    @pytest.mark.usefixtures("spark")
    def test_drop_cols(self, spark):
        cols = ["col_a", "col_b", "col_c", "col_d", "col_e"]
        data = [("a", "b", "c", "d", "e")]
        df = spark.createDataFrame(data, cols)
        dropped_cols = dataframe.drop_cols(["col_a", "col_c", "col_e"], df)
        expected_cols = ["col_b", "col_d"]
        expected_data = [("b", "d")]

        self.validate_values(dropped_cols, expected_cols, expected_data)
