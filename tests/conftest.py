import os
import pytest
import logging
from dotenv import load_dotenv

from pyspark.sql import SQLContext

from pyspark_tooling.spark import get_local_conf, get_sql_context


@pytest.fixture(scope="session", autouse=True)
def load_env():
    # if a .env file has been provided then load it
    try:
        load_dotenv()
    except Exception:
        print("client is not using a .env file")


@pytest.fixture(scope="session", autouse=True)
def configure_logs():
    """Prevent default logs"""
    logging.getLogger("py4j").setLevel(logging.INFO)
    logging.getLogger("pyspark").setLevel(logging.INFO)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("nose").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def spark(request) -> SQLContext:
    """Create a new spark sql context"""
    sc = get_sql_context(get_local_conf())
    request.addfinalizer(lambda: sc.sparkSession.stop())
    yield sc


@pytest.fixture(scope="session")
def postgres_credentials():
    """Retrieve the postgres credentials"""
    env_vars = ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"]

    for var in env_vars:
        if not os.getenv(var):
            raise Exception(
                f"Missing env var: {var}. Please look at the .env.example or read the README."
            )

    return {
        "host": os.environ["PGHOST"],
        "port": os.environ["PGPORT"],
        "database": os.environ["PGDATABASE"],
        "user": os.environ["PGUSER"],
        "password": os.environ["PGPASSWORD"],
    }
