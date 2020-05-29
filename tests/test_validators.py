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
class TestValidatorBaseClass(base.BaseTest):

    def test_string_validation(self):
        v = validators.Validator()

        a = str(uuid.uuid4())
        b = v.validate_str(a)
        assert b == a

        with pytest.raises(ValueError):
            v.validate_str(0.1)

        with pytest.raises(ValueError):
            v.validate_str("")      

        c = v.validate_str(None, allow_nulls=True)
        assert c is None

    def test_integer_validation(self):
        v = validators.Validator()

        a = random.randint(1, 100)
        b = v.validate_int(a)
        assert b == a

        with pytest.raises(ValueError):
            v.validate_int("foobar")

        with pytest.raises(ValueError):
            v.validate_int(0)

        c = v.validate_int(None, allow_nulls=True)
        assert c is None

        d = v.validate_int(0, allow_zero=True)
        assert d is 0

    def test_float_validation(self):
        v = validators.Validator()

        a = random.uniform(1.0, 100.0)
        b = v.validate_float(a)
        assert b == a

        with pytest.raises(ValueError):
            v.validate_float("foobar")

        with pytest.raises(ValueError):
            v.validate_float(0)

        c = v.validate_float(None, allow_nulls=True)
        assert c is None

        d = v.validate_float(0.0, allow_zero=True)
        assert d is 0.0

    def test_bool_validation(self):
        v = validators.Validator()

        a = random.choice([True, False])
        b = v.validate_bool(a)
        assert b == a

        with pytest.raises(ValueError):
            v.validate_bool("foobar")

        with pytest.raises(ValueError):
            v.validate_bool(0)

    def test_list_validation(self):
        v = validators.Validator()

        a = list(range(10))
        b = v.validate_list(a, of_type=int)
        assert b == a

        with pytest.raises(ValueError):
            v.validate_list(a + [None], of_type=int)

        with pytest.raises(ValueError):
            v.validate_list(a + ["string not allowed"], of_type=int)

        with pytest.raises(ValueError):
            v.validate_list("not a list")

        c = v.validate_list(None, allow_nulls=True)
        assert c is None


    def test_dictionary_validation(self):
        v = validators.Validator()

        a = {"foo": 1, "bar": 2}
        b = v.validate_dict(a, key_type=str, value_type=int, allow_nulls=False)
        assert b == a

        with pytest.raises(ValueError):
            c = a.update({"biz": None})
            v.validate_dict(c, key_type=str, value_type=int, allow_nulls=False)

        with pytest.raises(ValueError):
            c = a.update({"biz": "str"})
            v.validate_dict(c, key_type=str, value_type=int)


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
