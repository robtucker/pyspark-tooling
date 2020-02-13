#!/usr/bin/env python

import os

from setuptools import setup

THIS_DIR = os.path.dirname(__file__)

with open(os.path.join(THIS_DIR, "README.md")) as f:
    desc = f.read()

setup(
    name="pyspark-tooling",
    version="1.0.0",
    url="https://github.com/robtucker/pyspark-tooling.git",
    packages=["pyspark_tooling"],
    author="Robert Tucker",
    author_email="robert@coderlab.co.uk",
    description="Pyspark utility functions",
    long_description=desc,
    long_description_content_type="text/markdown",
    install_requires=[
        "boto3",
        "pandas",
        "pyspark>=2.4.0",
        "pytz>=2019.2",
        "Unidecode>=1.1.1",
        "PyYAML",
    ],
    include_package_data=True,
    platforms="any",
    zip_safe=False,
)
