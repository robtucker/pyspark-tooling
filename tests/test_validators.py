import pytest

from pyspark.sql import SQLContext
from pyspark.sql.types import (
    StructType,
    StructField,
    ArrayType,
    StringType,
    IntegerType,
)

from tests import base
from pyspark_tooling import validators
from pyspark_tooling.dataframe import to_tuples
from pyspark_tooling.exceptions import DataFrameException, SchemaException


# @pytest.mark.focus
class TestValidatorUtils(base.BaseTest):
    @pytest.mark.usefixtures("spark")
    def test_mismatching_values(self, spark: SQLContext):
        """Assert that mis-matching nullable booleans will raise an exception"""

        data = [("a", 1), ("b", 2), ("c", None)]

        actual_schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        df = spark.createDataFrame(data, actual_schema)

        incorrect_data = [("a", 1), ("b", 2), ("c", 3)]

        with pytest.raises(ValueError):
            validators.validate_values(df, actual_schema, incorrect_data)

    @pytest.mark.usefixtures("spark")
    def test_mismatching_row_count(self, spark: SQLContext):
        """Assert that mis-matching nullable booleans will raise an exception"""

        data = [("a", 1), ("b", 2), ("c", None)]

        actual_schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        df = spark.createDataFrame(data, actual_schema)

        incorrect_data = [("a", 1), ("b", 2)]

        with pytest.raises(DataFrameException):
            validators.validate_values(df, actual_schema, incorrect_data)

    @pytest.mark.usefixtures("spark")
    def test_mismatching_column_count(self, spark: SQLContext):
        """Assert that mis-matching nullable booleans will raise an exception"""

        data = [("a", 1), ("b", 2), ("c", None)]

        actual_schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        df = spark.createDataFrame(data, actual_schema)

        incorrect_data = [("a", 1, "x"), ("b", 2, "y"), ("c", None, "z")]

        with pytest.raises(DataFrameException):
            validators.validate_values(df, actual_schema, incorrect_data)

    @pytest.mark.usefixtures("spark")
    def test_recursive_struct_validation(self, spark: SQLContext):

        nested_nested_schema = StructType(
            [
                StructField("num_2", IntegerType()),
                StructField("arr_2", ArrayType(StringType())),
            ]
        )

        nested_schema = StructType(
            [
                StructField("num_1", IntegerType()),
                StructField("arr_1", ArrayType(StringType())),
                StructField("col_c", nested_nested_schema),
            ]
        )

        schema = StructType(
            [StructField("col_a", nested_schema), StructField("col_b", nested_schema)]
        )

        # rows are represented as tuples, whereas arrays are lists
        a = [
            (
                (1, ["a1", "b1", "c1"], (11, ["x1", "y1", "z1"])),
                (2, ["a2", "b2", "c2"], (12, ["x2", "y2", "z2"])),
            ),
            (
                (3, ["a3", "b3", "c3"], (13, ["x3", "y3", "z3"])),
                (4, ["a4", "b4", "c4"], (14, ["x4", "y4", "z4"])),
            ),
        ]

        # same as a but with one wrong val
        b = [
            (
                (1, ["a1", "b1", "c1"], (11, ["x1", "y1", "z1"])),
                (2, ["a2", "b2", "c2"], (12, ["x2", "y2", "z2"])),
            ),
            (
                (3, ["a3", "b3", "c3"], (13, ["x3", "y3", "WRONG VAL"])),
                (4, ["a4", "b4", "c4"], (14, ["x4", "y4", "z4"])),
            ),
        ]

        df = spark.createDataFrame(a, schema)

        print("res")
        print(to_tuples(df))
        # should be the exact specification defined by the to_tuples helper
        assert to_tuples(df) == a

        # should pass the validation helper
        validators.validate_values(df, schema, a)

        # should fail with an incorrect nested value
        with pytest.raises(ValueError):
            validators.validate_values(df, schema, b)

    @pytest.mark.usefixtures("spark")
    def test_mismatching_column_names(self, spark: SQLContext):
        """Assert that mis-matching nullable booleans will raise an exception"""

        data = [("a", 1), ("b", 2), ("c", None)]

        actual_schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        df = spark.createDataFrame(data, actual_schema)

        incorrect_schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("vals", IntegerType(), nullable=True),  # wrong name
            ]
        )

        with pytest.raises(SchemaException):
            validators.validate_schema(df, incorrect_schema)

    @pytest.mark.usefixtures("spark")
    def test_mismatching_column_types(self, spark: SQLContext):
        """Assert that mis-matching nullable booleans will raise an exception"""

        data = [("a", 1), ("b", 2), ("c", None)]

        actual_schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        df = spark.createDataFrame(data, actual_schema)

        incorrect_schema = StructType(
            [
                StructField("key", StringType(), nullable=True),
                StructField("value", StringType(), nullable=True),  # wrong type
            ]
        )

        with pytest.raises(SchemaException):
            validators.validate_schema(df, incorrect_schema)

    @pytest.mark.usefixtures("spark")
    def test_mismatching_nullables(self, spark: SQLContext):
        """Assert that mis-matching nullable booleans will raise an exception"""

        data = [("a", 1), ("b", 2), ("c", None)]

        actual_schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        df = spark.createDataFrame(data, actual_schema)

        incorrect_schema = StructType(
            [
                StructField("key", StringType(), nullable=True),  # wrong nullable bool
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        with pytest.raises(SchemaException):
            validators.validate_schema(df, incorrect_schema)
