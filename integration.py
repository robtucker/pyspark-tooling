import os
from grada_pyspark_utils.emr import EMR
from grada_pyspark_utils.timestamp import format_timestamp, utcnow


class IntegrationTest:
    def __init__(self):
        self.bucket = os.environ["INTEGRATION_TEST_BUCKET"]
        self.folder = f"landing_timestamp={format_timestamp(utcnow())}"
        self.s3_path = f"s3://{bucket}/{folder}"

    def pipeline(self):
        self.create_scripts()
        self.run_cluster()

    def create_scripts(self):
        self.copy_main_file()
        self.copy_code_bundle()
        self.copy_requirements()
        self.copy_bootstrap()

    def run_cluster():

        emr = EMR.from_defaults(
            cluster_name="integration_test_cluster",
            code_path=f"{s3_path}/main.py",
            log_uri=f"{s3_path}/logs",
            minimum_memory_in_gb=32,
            minimum_vcpu=12,
        )

        emr.run(sync=True)
