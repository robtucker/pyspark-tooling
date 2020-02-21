import os

from pyspark_tooling.logger import log


def get_parallellism():
    """The EMR cluster should provide information
    about how many partitions are available"""

    parallellism = int(os.environ["PYSPARK_DEFAULT_PARALLELLISM"])

    log.info("check pyspark parallellism", parallellism=parallellism)

    assert isinstance(parallellism, int)
    assert parallellism > 0
    return parallellism


def get_target(step="main"):

    s3_path = os.environ["S3_PATH"]

    log.info("check s3 path", path=s3_path)

    assert isinstance(s3_path, str)
    assert len(s3_path) > 0

    return f"{s3_path}/target/step={step}"
