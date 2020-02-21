import os
import uuid
import pandas as pd
from zipfile import ZipFile

from pyspark_tooling.filesystem import EmrFilesystem
from tests import base

CSV_DATA = """
foo,bar
a,1
b,2
c,3
"""

YAML_DATA = """
foo:
  - a: 1
  - b: 2
"""


SQL_DATA = """
SELECT *
FROM my_table
WHERE my_condition IS NOT NULL
"""


class TestFilesystem(base.BaseTest):
    """Test loading data from emr"""

    def test_load_local_yaml(self):
        filepath = f"./data/{str(uuid.uuid4())}/local.yaml"

        self.put_local(filepath, YAML_DATA)

        fs = EmrFilesystem(None, is_local=True)
        actual = fs.open(filepath)
        expected = {"foo": [{"a": 1}, {"b": 2}]}

        assert actual == expected
        self.wipe_folder("./data")

    def test_load_local_csv(self):
        filepath = f"./data/{str(uuid.uuid4())}/local.csv"

        self.put_local(filepath, CSV_DATA)

        fs = EmrFilesystem(None, is_local=True)
        actual = fs.open(filepath)

        tuples = [tuple(x) for x in actual.to_numpy()]
        expected = [("a", 1), ("b", 2), ("c", 3)]

        assert tuples == expected
        self.wipe_folder("./data")

    def test_load_local_txt(self):
        filepath = f"./data/{str(uuid.uuid4())}/local.txt"
        data = str(uuid.uuid4())

        self.put_local(filepath, data)

        fs = EmrFilesystem(None, is_local=True)
        actual = fs.open(filepath)

        assert actual == data
        self.wipe_folder("./data")

    def test_load_zip_csv(self):
        bucket = "./data"
        filename = "random.csv"

        actual = self.run_zip_test(bucket, filename, CSV_DATA)
        assert isinstance(actual, pd.core.frame.DataFrame)

        tuples = [tuple(x) for x in actual.to_numpy()]
        expected = [("a", 1), ("b", 2), ("c", 3)]

        assert tuples == expected

    def test_load_zip_yaml(self):
        bucket = "./data"
        filename = "random.yaml"
        actual = self.run_zip_test(bucket, filename, YAML_DATA)
        assert isinstance(actual, dict)
        expected = {"foo": [{"a": 1}, {"b": 2}]}

        assert actual == expected

    def test_load_zip_sql(self):
        bucket = "./data"
        filename = "random.sql"
        actual = self.run_zip_test(bucket, filename, SQL_DATA)
        assert isinstance(actual, str)

        assert actual == SQL_DATA

    def test_load_zip_txt(self):
        bucket = "./data"
        filename = "random.txt"
        data = f"{str(uuid.uuid4())}\n{str(uuid.uuid4())}"
        actual = self.run_zip_test(bucket, filename, data)
        assert isinstance(actual, str)
        assert actual == data

    def run_zip_test(self, bucket, filename: str, data):
        zip_folder = f"zip_test/{str(uuid.uuid4())}"
        zip_directory = os.path.join(bucket, zip_folder)

        filepath = f"{str(uuid.uuid4())}/{filename}"
        _, ext = os.path.splitext(filepath)

        local_filepath = os.path.join(zip_folder, filepath)
        zip_filepath = os.path.join(zip_directory, "result.zip")

        self.put_local(os.path.join(bucket, local_filepath), data)

        z = ZipFile(zip_filepath, mode="w")
        z.write(os.path.join(bucket, local_filepath), arcname=filepath)
        z.close()

        fs = EmrFilesystem(zipped_code_path=zip_filepath, is_local=False)
        actual = fs.open(filepath)
        self.wipe_folder(bucket)
        return actual

    def put_local(self, filepath: str, data):
        """Save a file locally"""
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        """Save a json file locally"""
        with open(filepath, "w") as f:
            f.write(data)
