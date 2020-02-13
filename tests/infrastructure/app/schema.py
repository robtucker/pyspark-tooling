from pyspark.sql.types import StructType, StructField, StringType, IntegerType

ID = "id"
GROUP = "group"
VALUE = "value"


def create_schema():
    return StructType(
        [
            StructField(ID, StringType()),
            StructField(GROUP, StringType()),
            StructField(VALUE, IntegerType()),
        ]
    )
