from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
from typing import List

from pyspark_tooling.dataframe import to_tuples
from pyspark_tooling.exceptions import DataFrameException, SchemaException


class Validator:
    """Validator base class"""

    def validate_str(self, string: str, allow_nulls=False):
        if allow_nulls and string is None:
            return string

        if not isinstance(string, str) or len(string) == 0:
            raise ValueError("not a valid str")
        return string

    def validate_int(self, integer: int, allow_zero: bool = False, allow_nulls=False):
        if allow_nulls and integer is None:
            return integer

        if not isinstance(integer, int):
            raise ValueError("not a valid int")

        if not allow_zero and (integer == 0):
            raise ValueError("int cannot be 0")

        return integer

    def validate_float(self, integer: int, allow_zero: bool = False, allow_nulls=False):
        if allow_nulls and integer is None:
            return integer

        if not isinstance(integer, float):
            raise ValueError("not a valid float")

        if not allow_zero and (integer == 0):
            raise ValueError("float cannot be 0")

        return integer

    def validate_numeric(
        self, number: int, allow_zero: bool = False, allow_nulls=False
    ):
        if allow_nulls and number is None:
            return number

        if not isinstance(number, int) and not isinstance(number, float):
            raise ValueError("not a valid int or float")

        if not allow_zero and ((number == 0) or (number == 0.0)):
            raise ValueError("number cannot be 0")

        return number

    def validate_bool(self, integer: int):
        if not isinstance(integer, bool):
            raise ValueError("not a valid bool")
        return integer

    def validate_list(self, lst: List[str], of_type=None, allow_nulls=False):
        if allow_nulls and lst is None:
            return lst
        if not isinstance(lst, (list, tuple)):
            raise ValueError("not a valid list or tuple")

        if of_type:
            for i in lst:
                if not isinstance(i, of_type):
                    raise ValueError(f"list item is not of type {of_type}")
        return lst

    def validate_dict(
        self, dictionary: dict, key_type=None, value_type=None, allow_nulls=False
    ):
        if allow_nulls and dictionary is None:
            return dictionary

        if not isinstance(dictionary, dict):
            raise ValueError("not a valid dict")

        if key_type:
            for i in dictionary.keys():
                if not isinstance(i, key_type):
                    raise ValueError(f"dict key is not of type {key_type}")

        if value_type:
            for i in dictionary.values():
                if not isinstance(i, value_type):
                    raise ValueError(f"dict value is not of type {value_type}")

        return dictionary


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


class Validator:
    """Validator base class"""

    def validate_str(self, s: str, allow_nulls=False):
        if allow_nulls and s is None:
            return s

        if not isinstance(s, str) or len(s) == 0:
            raise ValueError("not a valid str")
        return s

    def validate_int(self, i: int, allow_zero: bool = False, allow_nulls=False):
        if allow_nulls and i is None:
            return i

        if not isinstance(i, int):
            raise ValueError("not a valid int")

        if not allow_zero and (i == 0):
            raise ValueError("int cannot be 0")

        return i

    def validate_float(self, i: int, allow_zero: bool = False, allow_nulls=False):
        if allow_nulls and i is None:
            return i

        if not isinstance(i, float):
            raise ValueError("not a valid float")

        if not allow_zero and (i == 0):
            raise ValueError("float cannot be 0")

        return i

    def validate_numeric(self, i: int, allow_zero: bool = False, allow_nulls=False):
        if allow_nulls and i is None:
            return i

        if not isinstance(i, int) and not isinstance(i, float):
            raise ValueError("not a valid int or float")

        if not allow_zero and ((i == 0) or (i == 0.0)):
            raise ValueError("number cannot be 0")

        return i

    def validate_bool(self, i: int):
        if not isinstance(i, bool):
            raise ValueError("not a valid bool")
        return i

    def validate_list(self, l: List[str], of_type=None, allow_nulls=False):
        if allow_nulls and l is None:
            return l
        if not isinstance(l, (list, tuple)):
            raise ValueError("not a valid list or tuple")

        if of_type:
            for i in l:
                if not isinstance(i, of_type):
                    raise ValueError(f"list item is not of type {of_type}")
        return l

    def validate_dict(self, d: dict, key_type=None, value_type=None, allow_nulls=False):
        if allow_nulls and d is None:
            return d

        if not isinstance(d, dict):
            raise ValueError("not a valid dict")

        if key_type:
            for i in d.keys():
                if not isinstance(i, key_type):
                    raise ValueError(f"dict key is not of type {key_type}")

        if value_type:
            for i in d.values():
                if not isinstance(i, value_type):
                    raise ValueError(f"dict value is not of type {value_type}")

        return d
