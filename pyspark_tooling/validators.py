from pyspark.sql import DataFrame
from pyspark.sql.types import StructType

from grada_pyspark_utils.dataframe import to_tuples
from grada_pyspark_utils.exceptions import DataFrameException, SchemaException


def validate_values(
    df: DataFrame,
    expected_schema,
    expected_values: list,
    enforce_array_order=True,
    verbose=False,
):
    """Validate that the dataframe contains an exact list of rows and columns"""
    # validate the expected columns
    validate_schema(df, expected_schema, verbose=verbose)

    row_count = df.count()
    if row_count == 0:
        raise DataFrameException("DataFrame has 0 rows")

    if row_count != len(expected_values):
        raise DataFrameException(
            f"Incorrect number of rows: Received {row_count} - Expected: {len(expected_values)}"
        )

    res = to_tuples(df)
    col_count = len(res[0])
    for row_index, expected in enumerate(expected_values):
        actual = res[row_index]
        if verbose:
            print("Actual:")
            print(actual)
            print("Expected:")
            print(expected)
        # should have the same number of columns in each row
        if len(actual) != len(expected):
            raise DataFrameException(
                f"Incorrect number of columns: Received {len(actual)} - Expected: {len(expected)}"
            )

        for col_index in range(col_count):
            _recursive_validator(
                actual[col_index],
                expected[col_index],
                enforce_array_order=enforce_array_order,
            )


def _recursive_validator(a, e, enforce_array_order=True):
    """Recursively validate the actual data against the expected data"""
    # rows are tuples and therefore should be recursively validated
    if isinstance(a, tuple):
        # allow lists and tuples to be interchangeable
        if not isinstance(e, tuple) and not isinstance(e, list):
            raise ValueError(f"expected a {type(e)}. received {type(a)}")

        if len(a) != len(e):
            raise ValueError(
                f"Rows have mismatching lengths: Received {len(a)}. Expected {len(e)}"
            )
        for i in range(len(e)):
            _recursive_validator(a[i], e[i])
    # collected sets are lists and should be converted to sets
    elif isinstance(a, list):
        # allow lists and tuples to be interchangeable
        if not isinstance(e, list) and not isinstance(e, tuple):
            raise ValueError(f"expected a {type(e)}. received {type(a)}")

        if len(a) != len(e):
            raise ValueError(
                f"Arrays have mismatching lengths: Received {len(a)}. Expected {len(e)}"
            )
        # assert that the lists have the same elements
        # but not neccessarily in the same order
        if not enforce_array_order:
            # note that if the list members are rows we can still convert
            # them to sets because tuples are hashable in python
            if set(a) != set(e):
                raise ValueError(f"Received: {a}. Expected: {e}")
        else:
            # assert that the lists are identical including ordering
            for i in range(len(e)):
                _recursive_validator(a[i], e[i])
    else:
        if a != e:
            raise ValueError(f"Received: {a}. Expected: {e}")


def validate_schema(df: DataFrame, schema, verbose=False):
    """Confirm the dataframe matches an exact schema"""

    if isinstance(schema, list):
        if verbose:
            print("Actual schema:")
            print(df.columns)
            print("Expected schema:")
            print(schema)

        if df.columns != schema:
            raise SchemaException(
                f"Mismatching schema. Received: {str(df.columns)}. Expected: {str(schema)}."
            )
        else:
            # schema passes
            return

    elif not isinstance(schema, StructType):
        raise Exception(
            "expected schema must either be a list of column names or a valid pyspark StructType"
        )

    # this means the schema is a valid strict type
    cols = [f.name for f in schema.fields]
    if verbose:
        print("Actual schema:")
        print(df.columns)
        print("Expected schema:")
        print(cols)

    if len(df.columns) != len(schema):
        raise SchemaException(
            f"Schemas have different lengths. Received {len(df.columns)}. Expected {len(cols)}"
        )

    for i, actual in enumerate(df.schema):
        expected = schema[i]
        if actual != expected:
            raise SchemaException(
                f"Mismatching struct field. Received {actual}. Expected {expected}"
            )
