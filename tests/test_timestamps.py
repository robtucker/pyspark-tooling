import pytest
import pytz

from grada_pyspark_utils import dataframe, timestamp

from tests import base


# @pytest.mark.focus
class TestDataframeUtils(base.BaseTest):
    @pytest.mark.usefixtures("spark")
    def test_utc_timestamps(self, spark):

        t = timestamp.utcnow()
        e = timestamp.format_timestamp(t)

        data = [("a",), ("b",), ("c",)]
        raw = spark.createDataFrame(data, ["key"])

        df = timestamp.with_timestamp("val", t, raw)

        for row in dataframe.to_tuples(df):
            assert row[1] == e

    @pytest.mark.usefixtures("spark")
    def test_non_utc_timestamps(self, spark):
        pass
        au = pytz.timezone("Australia/Sydney")
        t1 = timestamp.utcnow()
        e = timestamp.format_timestamp(t1)
        t2 = t1.astimezone(au)

        data = [("a",), ("b",), ("c",)]
        raw = spark.createDataFrame(data, ["key"])

        df = timestamp.with_timestamp("val", t2, raw)

        for row in dataframe.to_tuples(df):
            assert row[1] == e
