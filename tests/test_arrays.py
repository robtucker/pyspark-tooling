import pytest

from pyspark.sql.types import StructType, StructField, StringType, ArrayType
import pyspark.sql.functions as F

from tests import base
from grada_pyspark_utils import arrays
from grada_pyspark_utils.dataframe import to_tuples, to_dicts


# @pytest.mark.focus
class TestArrayUtils(base.BaseTest):
    """Array utils represent generic transforms to array type columns"""

    @pytest.mark.usefixtures("spark")
    def test_length(self, spark):
        col = ["col_a"]
        data = [(["a", "b", "c"],), (["x", "y"],)]
        df = spark.createDataFrame(data, col)
        res = df.withColumn("col_length", arrays.length(df["col_a"]))
        expected = [3, 2]
        actual = to_dicts(res)

        assert [i["col_length"] for i in actual] == expected

    @pytest.mark.usefixtures("spark")
    def test_remove_empty_strings(self, spark):
        data = [(["foo", "bar", "", None],), (["", None],), (None,)]

        df = spark.createDataFrame(data, ["input"])
        res = df.withColumn("output", arrays.remove_empty_strings(F.col("input")))
        res.show()
        expected = [a + b for a, b in zip(data, [(["foo", "bar"],), ([],), ([],)])]

        assert to_tuples(res) == expected

    @pytest.mark.usefixtures("spark")
    def test_array_union(self, spark):
        cols = ["col_a", "col_b"]
        data = [
            (["a", "b", "c"], ["a", "d"]),
            (["f", "g", "h"], ["e", "d"]),
            (["q", "a", "c"], ["a", "q", "f"]),
            (["p", "o", "c"], ["r", "t", "c"]),
        ]
        df = spark.createDataFrame(data, cols)
        df_union = df.withColumn(
            "col_union", arrays.array_union(df["col_a"], df["col_b"])
        )
        expected = [
            ["a", "b", "c", "d"],
            ["f", "g", "h", "e", "d"],
            ["q", "a", "c", "f"],
            ["p", "o", "c", "r", "t"],
        ]
        actual = to_dicts(df_union)

        assert [i["col_union"] for i in actual] == expected

    @pytest.mark.usefixtures("spark")
    def test_array_intersection(self, spark):
        col = ["col_a", "col_b"]
        data = [
            (["a", "b", "c", "a"], ["a", "b"]),
            (["a", "a", "d", "e"], ["d"]),
            (["d", "b", "c", "a"], ["b", "c", "d"]),
            (["apple", "b", "a", "d"], ["apple", "d"]),
        ]
        df = spark.createDataFrame(data, col)
        df_intersection = df.withColumn(
            "col_intersection", arrays.array_intersection(df["col_a"], df["col_b"])
        )
        expected = [["a", "b"], ["d"], ["d", "b", "c"], ["apple", "d"]]
        actual = to_dicts(df_intersection)
        assert [i["col_intersection"] for i in actual] == expected

    @pytest.mark.usefixtures("spark")
    def test_merge_collected_sets(self, spark):
        """Merge collected sets into a set of unique values"""

        col_a = "col_a"
        col_b = "col_b"
        col_c = "col_c"

        data = [
            (["a", "b", "c", "d"], ["c", "d", "e"]),
            (["x", "y", "z"], ["x"]),
            (None, ["foo", "bar", "baz"]),
            (["random"], None),
            (None, None),
        ]

        expected = [
            set(["a", "b", "c", "d", "e"]),
            set(["x", "y", "z"]),
            set(["foo", "bar", "baz"]),
            set(["random"]),
            set(),
        ]

        schema = StructType(
            [
                StructField(col_a, ArrayType(StringType())),
                StructField(col_b, ArrayType(StringType())),
            ]
        )

        input_df = spark.createDataFrame(data, schema)
        output_df = input_df.select(
            arrays.merge_collected_sets(F.col(col_a), F.col(col_b)).alias(col_c)
        )

        res = [set(i[col_c]) for i in to_dicts(output_df)]

        assert res == expected
