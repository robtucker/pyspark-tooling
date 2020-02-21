import pytest
import psycopg2
import random
import string

from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from tests import base
from pyspark_tooling import io


# @pytest.mark.focus
class TestIOUtils(base.BaseTest):
    @pytest.mark.usefixtures("spark")
    def test_csv_happy_path(self, spark: SQLContext):
        data = [("a", 1), ("b", 2), ("c", None)]

        schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        df = spark.createDataFrame(data, schema)

        path = self._random_path()

        io.write_csv(path, df, schema=schema)

        res = io.read_csv(spark, path, schema=schema).orderBy("key")

        self.validate_values(res, schema, data)
        self.wipe_data_folder()

    @pytest.mark.usefixtures("spark")
    def test_csv_enforces_nullable(self, spark: SQLContext):
        """Confirm that non null columns are enforced"""
        data = [("a", 1), ("b", 2), ("c", None)]

        nullable_schema = StructType(
            [
                StructField("key", StringType(), nullable=True),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        strict_schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=False),
            ]
        )

        # create the dataframe without validating nullables
        df = spark.createDataFrame(data, nullable_schema)

        path = self._random_path()

        # write the dataframe without validating nullables
        io.write_csv(path, df)

        with pytest.raises(Exception):
            # should throw an error - "This field is not nullable, but got None"
            io.read_csv(spark, path, schema=strict_schema).collect()

    @pytest.mark.usefixtures("spark")
    def test_parquet_happy_path(self, spark: SQLContext):
        """Read and write parquet with fixed schema"""

        data = [("a", 1), ("b", 2), ("c", None)]

        schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        df = spark.createDataFrame(data, schema)

        path = self._random_path()

        io.write_parquet(path, df, schema)

        res = io.read_parquet(spark, path, schema=schema).orderBy("key")

        self.validate_values(res, schema, data)
        self.wipe_data_folder()

    @pytest.mark.usefixtures("spark")
    def test_parquet_enforces_nullable(self, spark: SQLContext):
        """Confirm that non null columns are enforced"""
        data = [("a", 1), ("b", 2), ("c", None)]

        nullable_schema = StructType(
            [
                StructField("key", StringType(), nullable=True),
                StructField("value", IntegerType(), nullable=True),
            ]
        )

        strict_schema = StructType(
            [
                StructField("key", StringType(), nullable=False),
                StructField("value", IntegerType(), nullable=False),
            ]
        )

        # create the dataframe without validating nullables
        df = spark.createDataFrame(data, nullable_schema)

        path = self._random_path()

        # write the dataframe without validating nullables
        io.write_parquet(path, df)

        with pytest.raises(Exception):
            # should throw an error - "This field is not nullable, but got None"
            io.read_parquet(spark, path, schema=strict_schema).collect()

    @pytest.mark.usefixtures("spark", "postgres_credentials")
    def test_jdbc_happy_path(self, spark, postgres_credentials):

        table = self.table_name()
        user = postgres_credentials["user"]
        password = postgres_credentials["password"]
        url = self.pyspark_jdbc_url(postgres_credentials)

        data = [("a", 1), ("b", 2), ("c", None)]

        df = spark.createDataFrame(data, ["key", "value"])

        io.write_jdbc(url, table, user, password, df)

        res = self.select_and_drop_table(postgres_credentials, table)

        actual = sorted([tuple(i) for i in res], key=lambda x: x[0])

        assert actual == data

    def pyspark_jdbc_url(self, postgres_credentials: dict):
        host = postgres_credentials["host"]
        port = postgres_credentials["port"]
        database = postgres_credentials["database"]
        return f"jdbc:postgresql://{host}:{port}/{database}"

    def select_and_drop_table(self, postgres_credentials: dict, table: str):
        data = self.select(postgres_credentials, f"SELECT * FROM {table}")
        # print('selected data', table, data)
        self.exec(postgres_credentials, f"DROP TABLE {table};")
        # print('dropped table', table)
        return data

    def select(self, postgres_credentials: dict, query: str):
        """Select rows from the database"""
        conn = self.get_conn(postgres_credentials)
        cur = conn.cursor()
        cur.execute(query)
        data = cur.fetchall()
        conn.close()
        cur.close()
        return data

    def exec(self, postgres_credentials: dict, query: str):
        conn = self.get_conn(postgres_credentials)
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        conn.close()
        cur.close()

    def get_conn(self, postgres_credentials: dict):
        return psycopg2.connect(**postgres_credentials)

    def table_name(self, count=10):
        return str(
            "".join([random.choice(string.ascii_letters) for _ in range(count)])
        ).upper()
