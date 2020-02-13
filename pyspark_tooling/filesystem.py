import os
import json
import yaml
import pandas as pd
from zipfile import ZipFile


class EmrFilesystem:
    """Interact with objects inside an emr cluster"""

    def __init__(self, zipped_code_path: str, is_local: bool = False):
        """Opening a file inside an emr cluster is not always easy.
        If your file is inside the zipped code bundle then it is
        likely that you will need to unzip the bundle first.
        Provide the path to the zipped code to the constructor."""
        self.zipped_code_path = zipped_code_path
        self.is_local = is_local

    def open(self, filepath: str, mode="r"):
        """In production-like environments, unzip the code bundle
        and open a file, or if local, then simply open the file"""
        _, ext = os.path.splitext(filepath)

        if not self.is_local:
            f = self.open_zip_file(filepath, mode=mode)
            return self.parse_file(f, ext, decode_bytes=True)

        # opening the file locally is easy
        with open(filepath, mode=mode) as f:
            return self.parse_file(f, ext)

    def parse_file(self, file, ext: str, decode_bytes=False):
        if ext == ".csv":
            return pd.read_csv(file)
        if ext == ".json":
            return json.load(file)
        if ext == ".yaml" or ext == ".yml":
            return yaml.safe_load(file)
        # treat the file as a normal text file
        if decode_bytes:
            return file.read().decode()
        return file.read()

    def open_zip_file(self, filepath: str, mode="r"):
        """Load a csv file from zipped up python code inside EMR"""
        with ZipFile(self.zipped_code_path, "r") as z:
            return z.open(filepath, mode=mode)
