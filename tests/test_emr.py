import pytest
import copy
import random
import uuid
from datetime import datetime
from dateutil.tz import tzlocal
from typing import List, Any
from unittest.mock import MagicMock, call

from tests import base
from pyspark_tooling.emr import Cluster, InfrastructureConfig, EMRException


class MockCluster(Cluster):
    def __init__(self, client: Any, cluster_id: str = None):
        super().__init__(
            cluster_name=str(uuid.uuid4()),
            bootstrap_script_paths=[str(uuid.uuid4())],
            logs_path=str(uuid.uuid4()),
            env_vars={"TEST": str(uuid.uuid4())},
            executor_cores=1,
            executor_memory_in_gb=1,
            yarn_memory_overhead_in_gb=1,
            driver_cores=1,
            driver_memory_in_gb=1,
            executor_count=1,
            default_parallellism=1,
            master_instance_type=str(uuid.uuid4()),
            polling_interval_in_secs=0.1,
        )
        self.client = client

        if not cluster_id:
            self.cluster_id = str(uuid.uuid4())
        else:
            self.cluster_id = cluster_id


# @pytest.mark.focus
class TestInfrastructureConfig(base.BaseTest):
    def test_single_master_calculation(self):
        # the instance profile we are expecting
        instance_tuple = ("r5.4xlarge", 16, 128)
        # if we create a config with a minimum of 100gb memory
        config = InfrastructureConfig(minimum_spark_memory_in_gb=100)
        # the actual minimum memory should include a 10% bonus for yarn
        assert config.total_minimum_memory == 110
        # this can be satisfied by a single r5.4xlarge master node
        self.validate_infra(config, instance_tuple)

    def test_multi_core_calculation(self):
        # the instance profile we are expecting
        instance_tuple = ("r5.24xlarge", 96, 768)
        # if we create a config with a minimum of 2000gb memory
        config = InfrastructureConfig(minimum_spark_memory_in_gb=2000)
        # the actual minimum memory should include a 10% bonus for yarn
        assert config.total_minimum_memory == 2200
        # this can be satisfied by a 3 r5.24xlarge nodes
        assert config.master_instance_count == 1
        assert config.core_instance_count == 2
        # validate the available memory and vcpu
        self.validate_infra(config, instance_tuple)

    def validate_infra(self, config: InfrastructureConfig, instance_tuple: int):
        assert config.instance_type == instance_tuple[0]
        instance_vcpus = instance_tuple[1]
        instance_memory = instance_tuple[2]

        args = config.to_dict()

        num_executors = args["executor_count"]
        num_instances = (
            args["master_instance_count"]
            + args["core_instance_count"]
            + args["task_instance_count"]
        )

        # validate that the memory adds up
        total_memory = num_instances * instance_memory
        spark_memory = num_executors * args["executor_memory_in_gb"]
        yarn_memory = num_executors * args["yarn_memory_overhead_in_gb"]
        dirver_memory = args["driver_memory_in_gb"]
        used_memory = spark_memory + yarn_memory + dirver_memory
        print("total_memory", total_memory)
        print("used_memory", used_memory)
        assert used_memory <= total_memory
        # its possible for some of the memory to be unused (5% at most)
        self.validate_is_within(used_memory, total_memory, 5)

        # validate that the vcpus add up
        total_vcpus = num_instances * instance_vcpus
        spark_vcpu = num_executors * args["executor_cores"]
        # there is one vcpu for the hadoop daemon on each instance
        used_vcpus = spark_vcpu + args["driver_cores"] + num_instances
        assert used_vcpus == total_vcpus

    def validate_is_within(self, actual, expected, percent_margin):
        """The actual must be within a certain percent of the expected amount"""
        perc = 100 - (actual / expected * 100)
        print("perc", perc)
        assert perc <= percent_margin


class TestCluster(base.BaseTest):
    """Test cluster functionality"""

    def test_start(self):
        """Test the cluster starts and the cluster id is properly set"""
        client = MagicMock()
        cluster_id = str(uuid.uuid4())
        response = self.mock_job_flow_response(cluster_id)
        client.run_job_flow.return_value = response
        cluster = MockCluster(client)
        # replace step definitions with a string id
        step_defs = str(uuid.uuid4())
        actual = cluster.start(step_defs)

        assert cluster.cluster_id == cluster_id
        assert actual == response

    def test_get_step_ids(self):
        """Test listing step ids"""
        client = MagicMock()
        id1 = str(uuid.uuid4())
        id2 = str(uuid.uuid4())

        client.list_steps.return_value = self.mock_list_steps([id1, id2])

        cluster = MockCluster(client)

        res = cluster.list_step_ids()

        # steps are returned in reverse order
        assert res == [id2, id1]

        client.list_steps.assert_called_with(ClusterId=cluster.cluster_id)

    def test_describe_step(self):
        step_id = str(uuid.uuid4())
        cluster_id = str(uuid.uuid4())
        expected = self.mock_describe_step(step_id)

        client = MagicMock()
        client.describe_step.return_value = expected

        cluster = MockCluster(client, cluster_id=cluster_id)
        actual = cluster.describe_step(step_id)

        client.describe_step.assert_called_with(ClusterId=cluster_id, StepId=step_id)

        assert actual == expected["Step"]

    def test_list_steps(self):
        cluster_id = str(uuid.uuid4())
        step_ids = [str(uuid.uuid4()) in range(random.randint(2, 6))]
        expected = self.mock_list_steps(step_ids)

        client = MagicMock()
        client.list_steps.return_value = expected

        cluster = MockCluster(client, cluster_id=cluster_id)
        actual = cluster.list_steps()

        client.list_steps.assert_called_with(ClusterId=cluster_id)

        assert actual == expected

    def test_list_step_ids(self):
        cluster_id = str(uuid.uuid4())
        step_ids = [str(uuid.uuid4()) in range(random.randint(2, 6))]
        reverse_step_ids = copy.deepcopy(step_ids)
        reverse_step_ids.reverse()
        expected = self.mock_list_steps(step_ids)

        client = MagicMock()
        client.list_steps.return_value = expected

        cluster = MockCluster(client, cluster_id=cluster_id)
        actual = cluster.list_step_ids()

        client.list_steps.assert_called_with(ClusterId=cluster_id)

        assert actual == reverse_step_ids

    def test_describe_cluster(self):
        cluster_id = str(uuid.uuid4())
        expected = self.mock_describe_cluster(cluster_id)

        client = MagicMock()
        client.describe_cluster.return_value = expected

        cluster = MockCluster(client, cluster_id=cluster_id)
        actual = cluster.describe_cluster()

        client.describe_cluster.assert_called_with(ClusterId=cluster_id)

        assert actual == expected["Cluster"]

    def test_wait_for_completed_step(self):
        """Test the cluster starts and the cluster id is properly set"""
        client = MagicMock()
        step_id = str(uuid.uuid4())
        cluster_id = str(uuid.uuid4())

        client.describe_step.side_effect = [
            self.mock_describe_step(step_id, state="PENDING"),
            self.mock_describe_step(step_id, state="PENDING"),
            self.mock_describe_step(step_id, state="RUNNING"),
            self.mock_describe_step(step_id, state="COMPLETED"),
        ]

        client.describe_cluster.side_effect = [
            self.mock_describe_cluster(cluster_id, state="STARTING"),
            self.mock_describe_cluster(cluster_id, state="BOOTSTRAPPING"),
        ]

        cluster = MockCluster(client, cluster_id=cluster_id)
        cluster.wait_for_step_to_complete(step_id)

        client.describe_step.assert_has_calls(
            [
                call(ClusterId=cluster_id, StepId=step_id),
                call(ClusterId=cluster_id, StepId=step_id),
                call(ClusterId=cluster_id, StepId=step_id),
                call(ClusterId=cluster_id, StepId=step_id),
            ]
        )

        client.describe_cluster.assert_has_calls(
            [call(ClusterId=cluster_id), call(ClusterId=cluster_id)]
        )

    def test_wait_for_failed_step(self):
        """Test the cluster starts and the cluster id is properly set"""
        client = MagicMock()
        step_id = str(uuid.uuid4())
        cluster_id = str(uuid.uuid4())

        client.describe_step.side_effect = [
            self.mock_describe_step(step_id, state="PENDING"),
            self.mock_describe_step(step_id, state="PENDING"),
            self.mock_describe_step(step_id, state="RUNNING"),
            self.mock_describe_step(step_id, state="FAILED"),
        ]

        client.describe_cluster.side_effect = [
            self.mock_describe_cluster(cluster_id, state="STARTING"),
            self.mock_describe_cluster(cluster_id, state="BOOTSTRAPPING"),
            self.mock_describe_cluster(cluster_id, state="TERMINATING"),
        ]

        cluster = MockCluster(client, cluster_id=cluster_id)

        with pytest.raises(EMRException):
            cluster.wait_for_step_to_complete(step_id)

        client.describe_step.assert_has_calls(
            [
                call(ClusterId=cluster_id, StepId=step_id),
                call(ClusterId=cluster_id, StepId=step_id),
                call(ClusterId=cluster_id, StepId=step_id),
                call(ClusterId=cluster_id, StepId=step_id),
            ]
        )

        client.describe_cluster.assert_has_calls(
            [call(ClusterId=cluster_id), call(ClusterId=cluster_id)]
        )

    def mock_job_flow_response(self, cluster_id: str):
        return {
            "JobFlowId": cluster_id,
            "ClusterArn": "arn:aws:elasticmapreduce:eu-west-1:120063026773:cluster/j-TNVNWB0TXDLI",
            "ResponseMetadata": self.mock_response_metadata(),
        }

    def mock_describe_cluster(
        self,
        cluster_id: str,
        state: str = "RUNNING",
        state_change_reason: str = "STEP_FAILURE",
    ):
        return {
            "Cluster": self.mock_cluster(
                cluster_id=cluster_id,
                state=state,
                state_change_reason=state_change_reason,
            ),
            "ResponseMetadata": self.mock_response_metadata(),
        }

    def mock_list_steps(self, step_ids: List[str]):
        return {
            "Steps": [self.mock_step(i) for i in step_ids],
            "ResponseMetadata": self.mock_response_metadata(),
        }

    def mock_describe_step(
        self, step_id: str, state="PENDING", action_on_failure="TERMINATE_CLUSTER"
    ):
        return {
            "Step": self.mock_step(
                step_id, state=state, action_on_failure=action_on_failure
            ),
            "ResponseMetadata": self.mock_response_metadata(),
        }

    def mock_step(
        self, id: str, state="PENDING", action_on_failure="TERMINATE_CLUSTER"
    ):
        return {
            "Id": id,
            "Name": f"Step {str(uuid.uuid4())}",
            "Config": {
                "Jar": "command-runner.jar",
                "Properties": {},
                "Args": ["state-pusher-script"],
            },
            "ActionOnFailure": action_on_failure,
            "Status": {
                "State": state,
                "StateChangeReason": {},
                "Timeline": {
                    "CreationDateTime": datetime(
                        2020, 2, 18, 14, 43, 37, 946000, tzinfo=tzlocal()
                    )
                },
            },
        }

    def mock_cluster(
        self,
        cluster_id: str,
        cluster_name: str = "unit_test_cluster",
        state: str = "RUNNING",
        state_change_reason: str = "STEP_FAILURE",
        bucket: str = str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
    ):
        return {
            "Id": cluster_id,
            "Name": cluster_name,
            "Status": {
                "State": state,
                "StateChangeReason": {
                    "Code": state_change_reason,
                    "Message": "Shut down as step failed",
                },
                "Timeline": {
                    "CreationDateTime": datetime(
                        2020, 2, 19, 14, 9, 40, 489000, tzinfo=tzlocal()
                    ),
                    "ReadyDateTime": datetime(
                        2020, 2, 19, 14, 12, 42, 442000, tzinfo=tzlocal()
                    ),
                },
            },
            "Ec2InstanceAttributes": {
                "Ec2SubnetId": f"subnet-{str(uuid.uuid4())}",
                "RequestedEc2SubnetIds": [
                    f"subnet-{str(uuid.uuid4())}",
                    f"subnet-{str(uuid.uuid4())}",
                ],
                "Ec2AvailabilityZone": "eu-west-1a",
                "RequestedEc2AvailabilityZones": [],
                "IamInstanceProfile": "EMR_EC2_DefaultRole",
                "EmrManagedMasterSecurityGroup": f"sg-{str(uuid.uuid4())}",
                "EmrManagedSlaveSecurityGroup": f"sg-{str(uuid.uuid4())}",
            },
            "InstanceCollectionType": "INSTANCE_FLEET",
            "LogUri": f"s3n://{bucket}/{str(uuid.uuid4())}/logs/",
            "ReleaseLabel": "emr-5.27.0",
            "AutoTerminate": False,
            "TerminationProtected": False,
            "VisibleToAllUsers": True,
            "Applications": [{"Name": "Spark", "Version": "2.4.4"}],
            "Tags": [{"Key": "Costcentre", "Value": "unit_test"}],
            "ServiceRole": "EMR_DefaultRole",
            "NormalizedInstanceHours": 32,
            "MasterPublicDnsName": f"{str(uuid.uuid4())}.compute.amazonaws.com",
            "Configurations": [
                {
                    "Classification": "hadoop-env",
                    "Configurations": [
                        {
                            "Classification": "export",
                            "Properties": {
                                "GEMFURYURL": "https://gemfury-url.com",
                                "PYSPARK_DEFAULT_PARALLELLISM": "20",
                                "RUN_ID": run_id,
                            },
                        }
                    ],
                    "Properties": {},
                },
                {
                    "Classification": "capacity-scheduler",
                    "Properties": {
                        "yarn.scheduler.capacity.resource-calculator": "org.apache.hadoop.yarn.util.resource.DominantResourceCalculator"
                    },
                },
                {
                    "Classification": "spark",
                    "Properties": {"maximizeResourceAllocation": "false"},
                },
                {
                    "Classification": "spark-defaults",
                    "Properties": {
                        "spark.default.parallelism": "20",
                        "spark.driver.cores": "5",
                        "spark.driver.maxResultSize": "38G",
                        "spark.driver.memory": "38G",
                        "spark.dynamicAllocation.enabled": "false",
                        "spark.executor.cores": "5",
                        "spark.executor.instances": "2",
                        "spark.executor.memory": "38G",
                        "spark.executor.memoryOverhead": "4G",
                    },
                },
                {
                    "Classification": "spark-env",
                    "Configurations": [
                        {
                            "Classification": "export",
                            "Properties": {
                                "GEMFURYURL": "https://gemfury-url.com",
                                "PYSPARK_DEFAULT_PARALLELLISM": "20",
                                "RUN_ID": run_id,
                            },
                        }
                    ],
                    "Properties": {},
                },
            ],
            "ScaleDownBehavior": "TERMINATE_AT_TASK_COMPLETION",
            "KerberosAttributes": {},
            "ClusterArn": "arn:aws:elasticmapreduce:eu-west-1:120063026773:cluster/j-1JIRMRGOU47H1",
            "StepConcurrencyLevel": 1,
        }

    def mock_response_metadata(self):
        return {
            "RequestId": str(uuid.uuid4()),
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "x-amzn-requestid": str(uuid.uuid4()),
                "content-type": "application/x-amz-json-1.1",
                "content-length": "283",
                "date": "Tue, 18 Feb 2020 17:09:19 GMT",
            },
            "RetryAttempts": 0,
        }
