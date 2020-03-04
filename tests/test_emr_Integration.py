import pytest
import os
import boto3
import uuid
from zipfile import ZipFile
from pyspark_tooling.emr import Cluster, ActionOnFailure
from tests.example.bootstrap import bootstrap_script


@pytest.mark.skip(reason="Integration test")
@pytest.mark.integration
class TestEMRIntegration:
    def test_emr_integration(self):
        self.bucket = os.environ["INTEGRATION_TEST_BUCKET"]
        self.run_id = str(uuid.uuid4())
        self.folder = f"pyspark_integration_test/{self.run_id}"
        self.s3_path = f"s3://{self.bucket}/{self.folder}"
        self.s3 = boto3.client("s3")
        self.local_path = "./tests/example"
        self._pipeline()

    def _pipeline(self):
        """EMR integration test pipeline"""
        # upload the scripts to s3
        self._create_scripts()

        # create a cluster definition
        cluster = self._get_emr()

        # run the emr scripts
        cluster.run(
            code_entrypoint_path=f"{self.s3_path}/main.py",
            code_bundle_path=f"{self.s3_path}/bundle.zip",
            action_on_failure=ActionOnFailure.Continue,
            synchronous=True,
        )

        custom_step = cluster.create_step_definition(
            code_entrypoint_path=f"{self.s3_path}/additional_step.py",
            code_bundle_path=f"{self.s3_path}/bundle.zip",
            step_name="Additional step",
            action_on_failure=ActionOnFailure.Continue,
        )

        # add the step into the cluster
        cluster.add_steps([custom_step])

        # wait for all steps to finish
        cluster.wait_for_steps_to_complete()

        # terminate the cluster
        cluster.terminate()

    def _create_scripts(self):
        """Upload all the required scripts to s3"""

        # write a zip file
        self._bundle_app()

        # the bootstrap script doesn't have access to the env vars
        # for this reason it must be treated as a template
        # such that secret values are filled at compile time
        script = bootstrap_script(self.s3_path)

        self.s3.put_object(
            Body=script.encode(), Bucket=self.bucket, Key=f"{self.folder}/bootstrap.sh"
        )

        self.s3.upload_file(
            f"{self.local_path}/main.py", self.bucket, f"{self.folder}/main.py"
        )

        self.s3.upload_file(
            f"{self.local_path}/main.py",
            self.bucket,
            f"{self.folder}/additional_step.py",
        )

        self.s3.upload_file(
            f"{self.local_path}/bundle.zip", self.bucket, f"{self.folder}/bundle.zip"
        )

        # also upload the requirements file
        self.s3.upload_file(
            f"{self.local_path}/requirements.txt",
            self.bucket,
            f"{self.folder}/requirements.txt",
        )

    def _bundle_app(self):
        """Bundle the application"""
        with ZipFile(f"{self.local_path}/bundle.zip", "w") as zf:
            for root, _, files in os.walk(f"{self.local_path}/app/"):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = filepath[len(self.local_path) + 1 :]
                    zf.write(filepath, arcname)

    def _get_emr(self):
        """Retrieve an emr infrastructure instance"""
        return Cluster.from_defaults(
            cluster_name="integration_test_cluster",
            bootstrap_script_paths=[f"{self.s3_path}/bootstrap.sh"],
            logs_path=f"{self.s3_path}/logs",
            minimum_memory_in_gb=32,
            minimum_vcpus=12,
            env_vars={"S3_PATH": self.s3_path, "RUN_ID": self.run_id},
            tags=[{"Key": "Costcentre", "Value": "integration_test"}],
            subnet_ids=os.environ["INTEGRATION_TEST_SUBNET_IDS"].split(","),
            ssh_key="emr_integration_test",
            keep_cluster_alive=True,
        )
