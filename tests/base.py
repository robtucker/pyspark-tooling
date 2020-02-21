import os
import shutil
import uuid

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from pyspark_tooling import validators
from pyspark_tooling.dataframe import to_dicts


class BaseTest:

    _original_col_name = "original"
    _transformed_col_name = "transformed"

    def validate_values(
        self,
        df: DataFrame,
        expected_cols,
        expected_values: list,
        verbose=False,
        enforce_array_order=True,
    ):
        validators.validate_values(
            df,
            expected_cols,
            expected_values,
            enforce_array_order=enforce_array_order,
            verbose=verbose,
        )

    def validate_schema(self, df: DataFrame, schema, verbose=False):
        """Confirm the dataframe matches an exact schema"""
        validators.validate_schema(df, schema, verbose=verbose)

    def validate_to_decimal_places(self, a: list, b: list, decimal_places=6):
        for a1, b1 in zip(a, b):
            assert round(a1, decimal_places) == round(b1, decimal_places)

    def wipe_folder(self, path: str):
        """Wipe the given folder"""
        shutil.rmtree(path)

    def wipe_data_folder(self):
        shutil.rmtree("./data")

    def _run_transform_test(self, spark, func, data, expected, verbose=False):
        """Helper to run a utils test using a standardized format"""

        raw = self._get_single_column_df(spark, data)

        df = raw.withColumn(
            self._transformed_col_name, func(F.col(self._original_col_name))
        )

        res = self._get_transformed_data(df, self._transformed_col_name)
        if verbose:
            print(res)
            print(expected)
        assert res == expected

    def _get_single_column_df(self, spark, data: list):
        """Convert an array of data into a dataframe with a single column"""
        return spark.createDataFrame(data, [self._original_col_name])

    def _get_transformed_data(self, df: DataFrame, col_name: str):
        return [i[col_name] for i in to_dicts(df)]

    def _validate_saved_file_count(
        self, path: str, min_file_count: int, extension=".parquet"
    ):
        """Validate that the number of files meets the minimum number of expected files"""
        files = self._list_files(path, extension=extension)
        assert len(files) >= min_file_count

    def _list_files(self, path: str, extension=".parquet"):
        return list(filter(lambda f: f.endswith(extension), os.listdir(path)))

    def _random_path(self):
        return "./data/" + str(uuid.uuid4())
