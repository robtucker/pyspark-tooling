#!/usr/bin/env python

import os

from setuptools import setup, find_packages

THIS_DIR = os.path.dirname(__file__)

with open(os.path.join(THIS_DIR, "README.md")) as f:
    desc = f.read()

setup(
    name="pyspark-tooling",
    version="1.0.0",
    url="https://github.com/robtucker/pyspark-tooling.git",
    download_url="https://github.com/robtucker/pyspark-tooling/archive/v_01.tar.gz",
    packages=find_packages(
        include=["pyspark_tooling", "pyspark_tooling.*"], exclude=["tests"]
    ),
    author="Robert Tucker",
    author_email="rob@coderlab.co.uk",
    description="Pyspark utility functions",
    long_description=desc,
    long_description_content_type="text/markdown",
    install_requires=[
        "boto3",
        "cytoolz",
        "gensim",
        "nltk",
        "pandas",
        "pyspark>=2.4.0",
        "pytz>=2019.2",
        "PyYAML",
        "structlog",
        "Unidecode>=1.1.1",
    ],
    include_package_data=True,
    platforms="any",
    keywords=["pyspark"],
    zip_safe=False,
)
