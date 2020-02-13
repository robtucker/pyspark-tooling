from pyspark import SparkConf
from pyspark.sql import SparkSession, SQLContext


def get_default_conf(app_name: str, parallellism: int = 200):
    """Retrieve the default spark config object"""
    conf = SparkConf().setAppName(app_name).set("spark.sql.session.timeZone", "UTC")

    if not isinstance(parallellism, int):
        raise ValueError("Parallellism must be an integer")

    conf.set("spark.sql.shuffle.partitions", parallellism)
    conf.set("spark.default.parallelism", parallellism)
    conf.set("spark.driver.memory", "2G")
    return conf


def get_local_conf(app_name: str = "local_testing", parallellism: int = 2):
    """Get a config that is suitable for local development"""
    conf = get_default_conf(app_name=app_name, parallellism=parallellism)
    conf.set("spark.driver.host", "localhost")
    conf.setMaster("local[*]")
    conf.set("spark.executorEnv.PYTHONHASHSEED", 0)
    return conf


def get_spark_session(conf: SparkConf):
    """Get a spark session from a given config"""
    assert isinstance(conf, SparkConf)
    return SparkSession.builder.config(conf=conf).getOrCreate()


def get_sql_context(conf: SparkConf):
    """Get an sql context from a given config"""
    sc = get_spark_session(conf)
    # it is essential to add the second spark session param
    # for some operations involving the underlying rdd
    return SQLContext(sparkContext=sc.sparkContext, sparkSession=sc)
