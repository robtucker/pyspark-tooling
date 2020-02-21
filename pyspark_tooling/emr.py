import boto3
import copy
import math
import os
import time
from datetime import datetime, timedelta
from dateutil.tz import tzutc
from enum import Enum
from typing import List, Dict, Any

from pyspark_tooling.validators import Validator
from pyspark_tooling.logger import log


# AWS default iam roles
EMR_ROLE = "EMR_DefaultRole"
EMR_EC2_ROLE = "EMR_EC2_DefaultRole"
AWS_DEFAULT_REGION = "eu-west-1"

# env vars that will be made availble inside the cluster
PYSPARK_DEFAULT_PARALLELLISM = "PYSPARK_DEFAULT_PARALLELLISM"
PYSPARK_PYTHON = "PYSPARK_PYTHON"


class ClusterState(Enum):
    Starting = "STARTING"
    Bootstrapping = "BOOTSTRAPPING"
    Running = "RUNNING"
    Waiting = "WAITING"
    Terminating = "TERMINATING"
    Terminated = "TERMINATED"
    TerminatedWithErrors = "TERMINATED_WITH_ERRORS"


class StepState(Enum):
    Pending = "PENDING"
    Running = "RUNNING"
    Continue = "CONTINUE"
    Completed = "COMPLETED"
    Cancelled = "CANCELLED"
    Failed = "FAILED"
    Interrupted = "INTERRUPTED"


class ActionOnFailure(Enum):
    TerminateJobFlow = "TERMINATE_JOB_FLOW"
    TerminateCluster = "TERMINATE_CLUSTER"
    Cancel = "CANCEL_AND_WAIT"
    Continue = "CONTINUE"


class MarketType(Enum):
    on_demand = "ON_DEMAND"
    spot = "SPOT"


class InstanceOptimization(Enum):
    general = "general"
    compute = "compute"
    memory = "memory"
    storage = "storage"


class EMRException(Exception):
    pass


class InfrastructureConfig(Validator):

    # a list of supported instances keyed by optimization-type
    # in order of size (ascending)
    supported_instances = {
        "general": [
            ("m5.xlarge", 4, 16),
            ("m5.2xlarge", 8, 32),
            ("m5.4xlarge", 16, 64),
            ("m5.8xlarge", 32, 128),
            ("m5.12xlarge", 48, 192),
            ("m5.16xlarge", 64, 256),
            ("m5.24xlarge", 96, 384),
        ],
        "compute": [
            ("c5.xlarge", 4, 8),
            ("c5.2xlarge", 8, 16),
            ("c5.4xlarge", 16, 32),
            ("c5.9xlarge", 36, 72),
            ("c5.12xlarge", 48, 96),
            ("c5.18xlarge", 72, 144),
            ("c5.24xlarge", 96, 192),
        ],
        "memory": [
            ("r5.xlarge", 4, 32),
            ("r5.2xlarge", 8, 64),
            ("r5.4xlarge", 16, 128),
            ("r5.12xlarge", 48, 384),
            ("r5.24xlarge", 96, 768),
        ],
        "storage": [
            ("i3.xlarge", 4, 30.5),
            ("i3.2xlarge", 8, 61),
            ("i3.4xlarge", 16, 122),
            ("i3.8xlarge", 32, 244),
            ("i3.16xlarge", 64, 488),
        ],
    }

    def __init__(
        self,
        # how much memory will be required to hold the RDDs?
        minimum_spark_memory_in_gb: int = 32,
        # optionally specify a minimum number of VCPUs
        minimum_vcpus: int = 6,
        # how many VCPUs should each executor have
        vcpus_per_executor: int = 5,
        # how many partitions should each VCPU have (min 1)
        partitions_per_vcpu: int = 2,
        # what fraction of an executors memory is given to yarn in range (0, 1)
        yarn_memory_overhead_fraction: int = 0.1,
        # what type of instance should be used
        instance_optimization: InstanceOptimization = InstanceOptimization.memory,
        # how many executors should be set aside for the driver (usually 1)
        num_executors_for_driver: int = 1,
    ):
        """Configure the EMR infrastructure using AWS best practices.
        Choose either a minimum amount of memory or a minimum number of VCPUS,
        and this config class will select your infrastructure for you, ensuring
        that both requirements will be met"""
        self.minimum_spark_memory_in_gb = self.validate_int(minimum_spark_memory_in_gb)
        self.minimum_vcpus = self.validate_int(minimum_vcpus)
        self.yarn_memory_overhead_fraction = self.validate_float(
            yarn_memory_overhead_fraction
        )
        self.vcpus_per_executor = self.validate_int(vcpus_per_executor)
        self.partitions_per_vcpu = self.validate_int(partitions_per_vcpu)
        self.instance_optimization = self.validate_str(instance_optimization.value)
        self.num_executors_for_driver = self.validate_int(num_executors_for_driver)

        # the total minimum memory includes the memory allocated to yarn
        yarn_memory = (
            self.minimum_spark_memory_in_gb * self.yarn_memory_overhead_fraction
        )
        self.total_minimum_memory = math.ceil(
            self.minimum_spark_memory_in_gb + yarn_memory
        )

        # calculate which instance type should be used
        self.set_instance_type(self.instance_optimization)
        self.calc_instance_counts()

    def set_instance_type(
        self, optimization: InstanceOptimization = InstanceOptimization.memory
    ):
        """Calculate which type of instance should be used based on the requirements"""
        instances = self.supported_instances[optimization]
        vcpus = [i[1] for i in instances]
        memory = [i[2] for i in instances]

        memory_index = 0
        vcpu_index = 0

        for i, v in enumerate(memory):
            memory_index = i
            if v > self.total_minimum_memory:
                break

        for i, v in enumerate(vcpus):
            vcpu_index = i
            if v > self.minimum_vcpus:
                break

        # pick an instance that satisfies both memory and vcpu requirements
        index = max([memory_index, vcpu_index])

        self.instance_type = instances[index][0]
        self.memory_per_instance = instances[index][2]
        self.vcpus_per_instance = instances[index][1]

    def calc_instance_counts(self, master_instance_count=1):
        """This is an opinionated configuration class that will assign
        a single master node and then will satisfy the rest of the memory and cpu
        requirements with core nodes. No task nodes will be used at all.

        TODO - support task nodes
        """
        # the minimum number of instances if we look at the memory requirements
        min_instances_memory = math.ceil(
            self.total_minimum_memory / self.memory_per_instance
        )

        # the minimum number of instances if we look at the vcpu requirements
        min_instances_vcpu = math.ceil(self.minimum_vcpus / self.vcpus_per_instance)

        self.total_instance_count = max([min_instances_memory, min_instances_vcpu, 1])
        self.master_instance_count = master_instance_count
        self.core_instance_count = self.total_instance_count - master_instance_count

    def effective_vcpus_per_instance(self):
        """The number of vcpus that come with the instance less
        one vcpu, which is reserved for hadoop daemons"""
        return self.vcpus_per_instance - 1

    def executors_per_instance(self):
        """Based on a fixed number of vcpus per executor,
        how many executors can we fit into this instance (rounded down)?"""
        return math.floor(self.effective_vcpus_per_instance() / self.vcpus_per_executor)

    def total_memory_per_executor(self):
        """Given how many executors we can fit into this instance,
        how much memory would each of these executors have?"""
        return self.memory_per_instance / self.executors_per_instance()

    def yarn_memory_per_executor(self):
        """How much of the total memory available to each
        executor in GB should be devoted to yarn"""
        return round(
            self.total_memory_per_executor() * self.yarn_memory_overhead_fraction
        )

    def spark_memory_per_executor(self):
        """How much of the total memory available to each
        executor in GB should be devoted to spark"""
        return round(
            self.total_memory_per_executor() * (1 - self.yarn_memory_overhead_fraction)
        )

    def total_executors(self):
        """The total number of executors across all instances"""
        return (
            self.executors_per_instance() * self.total_instance_count
        ) - self.num_executors_for_driver

    def total_vcpus(self):
        """The total number of vcpus for all instances
        minus the vcpus devoted to the hadoop daemon and the driver"""
        return self.total_executors() * self.vcpus_per_executor

    def partitions_per_executor(self):
        """How many paritions will each executor have?"""
        return self.partitions_per_vcpu * self.vcpus_per_executor

    def total_partitions(self):
        """The total number of partitions across all instances of this type"""
        return self.partitions_per_vcpu * self.total_vcpus()

    def memory_per_partition(self):
        """Based on the number of partitions in each executor, how much
        usable spark memory will be available to each partition?"""
        return self.spark_memory_per_executor() / self.partitions_per_executor()

    def driver_memory(self):
        """The amount of memory given to the driver is a determined
        by how many executors have been set aside to run the driver
        and how much memory each of those executors should have"""
        return self.num_executors_for_driver * self.spark_memory_per_executor()

    def driver_vcpus(self):
        """The amount of vcpus given to the driver is a determined
        by how many executors have been set aside to run the driver
        and how many vcpus each of those executors should have"""
        return self.num_executors_for_driver * self.vcpus_per_executor

    def to_dict(self):
        return {
            "executor_cores": self.vcpus_per_executor,
            "executor_memory_in_gb": self.spark_memory_per_executor(),
            "yarn_memory_overhead_in_gb": self.yarn_memory_per_executor(),
            "driver_cores": self.driver_vcpus(),
            "driver_memory_in_gb": self.driver_memory(),
            "executor_count": self.total_executors(),
            "default_parallellism": self.total_partitions(),
            "master_instance_type": self.instance_type,
            "master_instance_count": self.master_instance_count,
            "core_instance_type": self.instance_type,
            "core_instance_count": self.core_instance_count,
            "task_instance_type": self.instance_type,
            "task_instance_count": 0,
        }


class Cluster(Validator):
    """Class representing an EMR cluster"""

    def __init__(
        self,
        # name of the cluster as it appears in aws console
        cluster_name: str,
        # list of s3 paths to bootstrap scripts
        bootstrap_script_paths: List[str],
        # s3 path where the output logs should be written
        logs_path: str,
        # a dictionary of environment variables that will be passed into the emr cluster
        env_vars: Dict[str, str],
        # the number of vcpus assigned to each executor
        executor_cores: int,
        # the memory given to each executor not including the fraction given to yarn
        executor_memory_in_gb: int,
        # the fraction of each executor's total memory devoted to yarn
        yarn_memory_overhead_in_gb: int,
        # the number of vcpus assigned to the driver
        driver_cores: int,
        # the amount of memory devoted to the driver in gb
        driver_memory_in_gb: int,
        # the total number of executors (make sure you calculated this correctly)
        executor_count: int,
        # the default number of partitions
        default_parallellism: int,
        # the type of ec2 instance used for master nodes (required)
        master_instance_type: str,
        # the number of master nodes
        master_instance_count: int = 1,
        # the type of ec2 instance used for core nodes
        core_instance_type: str = None,
        # the number of core nodes
        core_instance_count: int = None,
        # the type of ec2 instance used for task nodes
        task_instance_type: str = None,
        # the number of task nodes
        task_instance_count: int = None,
        # should the cluster be terminated on successfull completion?
        keep_cluster_alive: bool = False,
        # do not set this to true unless you know what you are doing
        maximise_resource_allocation: bool = False,
        # do not set this to true unless you know what you are doing
        dynamic_allocation_enabled: bool = False,
        # the amount of time to wait for the infrastructure to boot up
        infrastructure_timeout_duration_in_mins: int = 15,
        # the type of infrastructure (ie. on demand or spot instance)
        market_type: MarketType = MarketType.spot,
        # for spot instance the percentage of demand price you are willing to pay
        bid_price_as_percentage_of_on_demand: int = 80,
        # the role used to create the emr cluster
        job_flow_role: str = EMR_EC2_ROLE,
        # the role used inside the emr cluster to run the job
        service_role: str = EMR_ROLE,
        # a list of applications that AWS must make available inside the cluster
        applications: List[dict] = [{"Name": "Spark"}],
        # custom tags used for cost allocation etc
        tags: List[Dict[str, Any]] = [],
        # custom jars required during execution
        jars: List[str] = None,
        # deploy in client or cluster mode
        deploy_mode: str = "cluster",
        # the fraction of memory given to spark (the rest is "user" memory)
        spark_memory_fraction: float = 0.75,
        # of the memory given to spark, how much is reserved for storage
        spark_memory_storage_fraction: float = 0.5,
        # the default aws region
        region_name=AWS_DEFAULT_REGION,
        # the ec2 key for the spot instance
        ssh_key: str = None,
        # a list of subnet ids
        subnet_ids: List[str] = None,
        # a logger instance
        logger: Any = None,
        # polling interval in seconds
        polling_interval_in_secs: int = 30,
        # should this cluster only be visible to the user who started it
        visible_to_all_users: bool = True,
        # which version of emr should be used
        release_label: str = "emr-5.27.0",
        # which yarn scheduler should be used
        yarn_scheduler: str = "org.apache.hadoop.yarn.util.resource.DominantResourceCalculator",
        # extra java options
        extraJavaOptions: str = "-XX:+UseParallelGC",
        # path where pyspark will look for the python executable
        pyspark_python_path: str = "/usr/bin/python3",
        # any other args to pass on to boto3
        **boto_kwargs,
    ):
        self.cluster_name = self.validate_str(cluster_name)
        self.bootstrap_script_paths = self.validate_list(
            bootstrap_script_paths, of_type=str
        )
        self.logs_path = self.validate_str(logs_path)
        self.env_vars = self.validate_dict(env_vars, key_type=str, value_type=str)
        self.executor_cores = self.validate_int(executor_cores)
        self.executor_memory_in_gb = self.validate_int(executor_memory_in_gb)
        self.yarn_memory_overhead_in_gb = self.validate_int(yarn_memory_overhead_in_gb)
        self.driver_cores = self.validate_int(driver_cores)
        self.driver_memory_in_gb = self.validate_int(driver_memory_in_gb)
        self.executor_count = self.validate_int(executor_count)
        self.default_parallellism = self.validate_int(default_parallellism)
        self.maximise_resource_allocation = self.validate_bool(
            maximise_resource_allocation
        )
        self.dynamic_allocation_enabled = self.validate_bool(dynamic_allocation_enabled)
        self.master_instance_type = self.validate_str(master_instance_type)
        self.master_instance_count = self.validate_int(master_instance_count)
        self.core_instance_type = self.validate_str(
            core_instance_type, allow_nulls=True
        )
        self.core_instance_count = self.validate_int(
            core_instance_count, allow_zero=True, allow_nulls=True
        )
        self.task_instance_type = self.validate_str(
            task_instance_type, allow_nulls=True
        )
        self.task_instance_count = self.validate_int(
            task_instance_count, allow_zero=True, allow_nulls=True
        )
        self.keep_cluster_alive = self.validate_bool(keep_cluster_alive)
        self.infrastructure_timeout_duration_in_mins = self.validate_int(
            infrastructure_timeout_duration_in_mins
        )
        self.market_type = self.validate_str(market_type.value)
        self.bid_price_as_percentage_of_on_demand = self.validate_int(
            bid_price_as_percentage_of_on_demand
        )
        self.job_flow_role = self.validate_str(job_flow_role)
        self.service_role = self.validate_str(service_role)
        self.applications = self.validate_list(applications)
        self.tags = self.validate_list(tags)
        self.jars = self.validate_list(jars, allow_nulls=True)
        self.deploy_mode = self.validate_str(deploy_mode)
        self.spark_memory_fraction = spark_memory_fraction
        self.spark_memory_storage_fraction = spark_memory_storage_fraction

        self.ssh_key = self.validate_str(ssh_key, allow_nulls=True)
        self.subnet_ids = self.validate_list(subnet_ids, allow_nulls=True)
        self.polling_interval_in_secs = self.validate_numeric(polling_interval_in_secs)

        self.visible_to_all_users = self.validate_bool(visible_to_all_users)
        self.release_label = self.validate_str(release_label)
        self.yarn_scheduler = self.validate_str(yarn_scheduler)
        self.extraJavaOptions = self.validate_str(extraJavaOptions, allow_nulls=True)
        self.pyspark_python_path = pyspark_python_path

        if not os.getenv("AWS_DEFAULT_REGION") and not boto_kwargs.get("region_name"):
            boto_kwargs["region_name"] = region_name

        self.session = boto3.Session(**boto_kwargs)
        self.client = self.session.client("emr")
        self.logger = logger

        if not self.logger:
            self.logger = log

        self.terminated_states = [
            ClusterState.Terminating,
            ClusterState.Terminated,
            ClusterState.TerminatedWithErrors,
        ]

    @staticmethod
    def from_defaults(
        cluster_name: str,
        logs_path: str,
        bootstrap_script_paths: List[str],
        env_vars: dict = {},
        minimum_spark_memory_in_gb: int = 32,
        minimum_vcpus: int = 4,
        vcpus_per_executor: int = 5,
        partitions_per_vcpu: int = 2,
        yarn_memory_overhead_fraction: int = 0.1,
        instance_optimization: InstanceOptimization = InstanceOptimization.memory,
        num_executors_for_driver: int = 1,
        keep_cluster_alive: bool = False,
        market_type: MarketType = MarketType.spot,
        bid_price_as_percentage_of_on_demand: int = 80,
        job_flow_role: str = EMR_EC2_ROLE,
        service_role: str = EMR_ROLE,
        tags: dict = {},
        ssh_key: str = None,
        subnet_ids: List[str] = None,
    ):
        """This function is a reduced subset of the arguments
        you can pass in to the main constructor, such that
        the difficult calculations have been done for you"""

        # perform the default calculations
        config = InfrastructureConfig(
            minimum_spark_memory_in_gb=minimum_spark_memory_in_gb,
            minimum_vcpus=minimum_vcpus,
            vcpus_per_executor=vcpus_per_executor,
            partitions_per_vcpu=partitions_per_vcpu,
            yarn_memory_overhead_fraction=yarn_memory_overhead_fraction,
            instance_optimization=instance_optimization,
            num_executors_for_driver=num_executors_for_driver,
        )

        return Cluster(
            cluster_name=cluster_name,
            bootstrap_script_paths=bootstrap_script_paths,
            logs_path=logs_path,
            env_vars=env_vars,
            keep_cluster_alive=keep_cluster_alive,
            market_type=market_type,
            bid_price_as_percentage_of_on_demand=bid_price_as_percentage_of_on_demand,
            job_flow_role=job_flow_role,
            service_role=service_role,
            tags=tags,
            ssh_key=ssh_key,
            subnet_ids=subnet_ids,
            # add the infrastructure config
            **config.to_dict(),
        )

    def run(
        self,
        code_entrypoint_path: str,
        code_bundle_path: str,
        synchronous=False,
        action_on_failure: ActionOnFailure = ActionOnFailure.TerminateCluster,
    ):
        """Create a new cluster and run the job flow"""

        # create the default step definitions
        step_defs = self.get_default_steps(
            code_entrypoint_path, code_bundle_path, action_on_failure=action_on_failure
        )
        # start a new job flow
        res = self.start(step_defs=step_defs)

        self.logger.info("Successfully started new cluster", cluster_id=self.cluster_id)

        # if the executing process doesn't need to wait
        if not synchronous:
            # then we can immediately return the result
            return res

        self.logger.info("Waiting for steps to complete", cluster_id=self.cluster_id)

        return self.wait_for_steps_to_complete()

    def start(self, step_defs: List[Any]):
        """Create a new EMR cluster"""

        self.validate_str(self.cluster_name)

        args = {
            "Name": self.cluster_name,
            "LogUri": self.logs_path,
            "ReleaseLabel": self.release_label,
            "Instances": self.get_infrastructure_config(),
            "Steps": step_defs,
            "Configurations": self.get_emr_config(),
            "Applications": self.applications,
            "BootstrapActions": self.get_bootstrap_actions(),
            "VisibleToAllUsers": self.visible_to_all_users,
            "JobFlowRole": self.job_flow_role,
            "ServiceRole": self.service_role,
            "Tags": self.tags,
        }
        self.logger.info("Starting cluster from configuration", args=args)
        # start the cluster
        res = self.client.run_job_flow(**args)
        # print("run_job_flow", res)
        # extract the id of this cluster from the response
        self.cluster_id = res["JobFlowId"]
        # ensure the cluster id is a valid string
        self.validate_cluster_id()
        return res

    def add_steps(self, step_defs: List[Any]):
        """Submit the constructed job flow steps to EMR for execution"""
        self.validate_cluster_id()
        self.validate_list(step_defs, of_type=dict)

        steps_kwargs = dict(JobFlowId=self.cluster_id, Steps=step_defs)
        self.logger.info(
            "Add new job flow steps", cluster_id=self.cluster_id, steps=step_defs
        )

        return self.client.add_job_flow_steps(**steps_kwargs)["StepIds"]

    def list_steps(self):
        """Thin wrapper over boto3 list_steps"""
        self.validate_cluster_id()
        return self.client.list_steps(ClusterId=self.cluster_id)

    def list_step_ids(self):
        """Retrieve a list of steps in execution order"""
        steps = self.list_steps()
        # print("steps", steps)
        res = [i["Id"] for i in steps["Steps"]]
        res.reverse()
        return res

    def describe_cluster(self):
        """Thin wrapper over boto3 describe_cluster"""
        self.validate_cluster_id()
        res = self.client.describe_cluster(ClusterId=self.cluster_id)
        # print("describe_cluster", res)
        return res["Cluster"]

    def describe_step(self, step_id):
        """Thin wrapper over boto3 describe_step"""
        self.validate_cluster_id()
        self.validate_str(step_id)
        res = self.client.describe_step(ClusterId=self.cluster_id, StepId=step_id)
        # print("describe step", res)
        return res["Step"]

    def terminate(self):
        """Thin wrapper over boto3 terminate_job_flows"""
        self.validate_cluster_id()
        return self.client.terminate_job_flows(JobFlowIds=[self.cluster_id])

    def wait_for_steps_to_complete(self):
        """Wait for every step of the job to complete, one by one."""
        step_ids = self.list_step_ids()

        for step_id in step_ids:
            self.logger.info(f"Waiting for step ID {step_id} to complete...")
            self.wait_for_step_to_complete(step_id)

    def wait_for_step_to_complete(self, step_id: str):
        """Wait for step with the given ID to complete"""
        self.validate_cluster_id()
        while True:
            time.sleep(self.polling_interval_in_secs)

            step = self.describe_step(step_id)
            step_state = _get_state(step)
            if StepState(step_state) == StepState.Pending:
                cluster = self.describe_cluster()
                cluster_state = _get_state(cluster)
                reason = _get_reason(cluster)
                reason_desc = (": %s" % reason) if reason else ""

                self.logger.info(
                    f"PENDING (cluster is {cluster_state}{reason_desc})",
                    cluster_id=self.cluster_id,
                    step_id=step_id,
                )
                continue

            elif StepState(step_state) == StepState.Running:
                time_running_desc = ""

                start = step["Status"]["Timeline"].get("StartDateTime")
                if start:
                    time_running_desc = " for %s" % strip_microseconds(
                        _boto3_now() - start
                    )

                self.logger.info(
                    f"RUNNING {time_running_desc}",
                    cluster_id=self.cluster_id,
                    step_id=step_id,
                )
                continue

            # we're done, will return at the end of this
            elif StepState(step_state) == StepState.Completed:
                self.logger.info(
                    "COMPLETED", cluster_id=self.cluster_id, step_id=step_id
                )
                return
            else:
                # step has failed somehow. *reason* seems to only be set
                # when job is cancelled (e.g. 'Job terminated')
                reason = _get_reason(step)

                self.logger.info(
                    f"Step failure: {step_state} {reason}", step_state=step_state
                )

                # print cluster status; this might give more context
                # why step didn't succeed
                cluster = self.describe_cluster()
                cluster_status = _get_state(cluster)
                reason = _get_reason(cluster)
                self.logger.info(
                    f"Cluster has status {cluster_status} {reason}",
                    cluster_id=self.cluster_id,
                    step_id=step_id,
                )

                if ClusterState(cluster_status) in self.terminated_states:
                    # # was it caused by IAM roles?
                    # self.check_for_missing_default_iam_roles(self.cluster_id)
                    pass

            if StepState(step_state) == StepState.Failed:
                self.logger.info(f"Step {step_id} failed")

            raise EMRException("step failed")

    def get_env_vars(self):
        """Retrieve the env vars that will be passed in to the EMR cluster"""

        required_env_vars = {
            # the pyspark parallellism env var is a required env
            # this can be used when constructing the spark context
            PYSPARK_DEFAULT_PARALLELLISM: str(self.default_parallellism),
            # the pyspark
            PYSPARK_PYTHON: self.pyspark_python_path,
        }
        # merge the required env vars with the ones supplied by the user
        res = copy.deepcopy(required_env_vars)
        res.update(self.env_vars)
        return res

    def get_default_steps(
        self,
        code_entrypoint_path: str,
        code_bundle_path: str,
        action_on_failure: ActionOnFailure = ActionOnFailure.TerminateCluster,
    ):
        """Return a list of default step definitions"""
        return [
            self.create_enable_debugging_step(),
            self.create_step_definition(
                code_entrypoint_path,
                code_bundle_path,
                action_on_failure=action_on_failure,
            ),
        ]

    def create_enable_debugging_step(self):
        """Set up the EMR logs output"""
        return {
            "Name": "Enable Debugging",
            "ActionOnFailure": ActionOnFailure.TerminateCluster.value,
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["state-pusher-script"],
            },
        }

    def create_step_definition(
        self,
        code_entrypoint_path: str,
        code_bundle_path: str,
        step_name: str = "Execution step",
        action_on_failure: ActionOnFailure = ActionOnFailure.TerminateCluster,
    ):
        """Retrieve a step definition for the given python files"""

        self.validate_str(code_entrypoint_path)
        self.validate_str(code_bundle_path)

        command = [
            "spark-submit",
            "--deploy-mode",
            self.deploy_mode,
            "--py-files",
            code_bundle_path,
            code_entrypoint_path,
        ]

        if self.jars:
            command.extend(["--jars", self.jars])

        return {
            "Name": step_name,
            "ActionOnFailure": action_on_failure.value,
            "HadoopJarStep": {"Jar": "command-runner.jar", "Args": command},
        }

    def get_bootstrap_actions(self):
        """Create a list of bootstrap actions"""
        return [
            {
                "Name": f"Bootstrap script {i.split('/')[-1]}",
                "ScriptBootstrapAction": {"Path": i},
            }
            for i in self.bootstrap_script_paths
        ]

    def get_infrastructure_config(self):
        """Choose the relevant instance config"""
        if self.market_type == MarketType.spot.value:
            return self.get_spot_config()
        return self.get_on_demand_config()

    def get_on_demand_config(self):
        """The on demand config is the simplest infrastructure config to use"""
        res = {
            "KeepJobFlowAliveWhenNoSteps": self.keep_cluster_alive,
            "InstanceGroups": [
                {
                    "Name": "Master",
                    "InstanceCount": self.master_instance_count,
                    "InstanceType": self.master_instance_type,
                    "Market": "ON_DEMAND",
                    "InstanceRole": "MASTER",
                }
            ],
        }
        if int(self.core_instance_count or 0) > 0:
            res["InstanceGroups"].append(
                {
                    "Name": "Core",
                    "InstanceCount": self.core_instance_count,
                    "InstanceType": self.core_instance_type,
                    "Market": "ON_DEMAND",
                    "InstanceRole": "CORE",
                }
            )
        if int(self.task_instance_count or 0) > 0:
            res["InstanceGroups"].append(
                {
                    "Name": "Task",
                    "InstanceCount": self.task_instance_count,
                    "InstanceType": self.task_instance_type,
                    "Market": "ON_DEMAND",
                    "InstanceRole": "TASK",
                }
            )
        return res

    def get_spot_config(self):
        """The spot config should reduce the cost of running emr quite drastically.
        Set the bid price as a percentage of demand price accordingly"""
        res = {
            "InstanceFleets": [
                {
                    "Name": "Master",
                    "InstanceFleetType": "MASTER",
                    "TargetSpotCapacity": self.master_instance_count,
                    "InstanceTypeConfigs": [
                        {
                            "BidPriceAsPercentageOfOnDemandPrice": self.bid_price_as_percentage_of_on_demand,
                            "InstanceType": self.master_instance_type,
                            "WeightedCapacity": 1,
                        }
                    ],
                    "LaunchSpecifications": {
                        "SpotSpecification": {
                            "TimeoutDurationMinutes": self.infrastructure_timeout_duration_in_mins,
                            "TimeoutAction": "SWITCH_TO_ON_DEMAND",
                        }
                    },
                }
            ],
            "KeepJobFlowAliveWhenNoSteps": self.keep_cluster_alive,
        }

        if int(self.core_instance_count or 0) > 0:
            res["InstanceFleets"].append(
                {
                    "Name": "Core",
                    "InstanceFleetType": "CORE",
                    "TargetSpotCapacity": self.core_instance_count,
                    "InstanceTypeConfigs": [
                        {
                            "BidPriceAsPercentageOfOnDemandPrice": self.bid_price_as_percentage_of_on_demand,
                            "InstanceType": self.core_instance_type,
                            "WeightedCapacity": 1,
                        }
                    ],
                    "LaunchSpecifications": {
                        "SpotSpecification": {
                            "TimeoutDurationMinutes": self.infrastructure_timeout_duration_in_mins,
                            "TimeoutAction": "SWITCH_TO_ON_DEMAND",
                        }
                    },
                }
            )

        if int(self.task_instance_count or 0) > 0:
            res["InstanceFleets"].append(
                {
                    "Name": "Task",
                    "InstanceFleetType": "TASK",
                    "TargetSpotCapacity": self.task_instance_count,
                    "InstanceTypeConfigs": [
                        {
                            "BidPriceAsPercentageOfOnDemandPrice": self.bid_price_as_percentage_of_on_demand,
                            "InstanceType": self.task_instance_type,
                            "WeightedCapacity": 1,
                        }
                    ],
                    "LaunchSpecifications": {
                        "SpotSpecification": {
                            "TimeoutDurationMinutes": self.infrastructure_timeout_duration_in_mins,
                            "TimeoutAction": "SWITCH_TO_ON_DEMAND",
                        }
                    },
                }
            )

        if self.subnet_ids:
            res["Ec2SubnetIds"] = self.subnet_ids

        if self.ssh_key:
            res["Ec2KeyName"] = self.ssh_key

        return res

    def get_emr_config(self):
        """The configuration dict for emr itself"""
        return [
            {
                "Classification": "hadoop-env",
                "Configurations": [
                    {
                        "Classification": "export",
                        "Configurations": [],
                        "Properties": self.get_env_vars(),
                    }
                ],
                "Properties": {},
            },
            {
                "Classification": "capacity-scheduler",
                "Properties": {
                    "yarn.scheduler.capacity.resource-calculator": self.yarn_scheduler
                },
            },
            {
                "Classification": "spark",
                "Properties": {
                    "maximizeResourceAllocation": str(
                        self.maximise_resource_allocation
                    ).lower()
                },
            },
            {
                "Classification": "spark-defaults",
                "Properties": self.get_spark_defaults(),
            },
            {
                "Classification": "spark-env",
                "Configurations": [
                    {
                        "Classification": "export",
                        "Configurations": [],
                        "Properties": self.get_env_vars(),
                    }
                ],
                "Properties": {},
            },
        ]

    def get_spark_defaults(self):
        """Configuration values used in tuning spark jobs"""
        return {
            "spark.default.parallelism": str(self.default_parallellism),
            "spark.driver.memory": f"{self.driver_memory_in_gb}G",
            "spark.driver.cores": str(self.driver_cores),
            "spark.driver.maxResultSize": f"{math.floor(self.driver_memory_in_gb)}G",
            "spark.dynamicAllocation.enabled": str(
                self.dynamic_allocation_enabled
            ).lower(),
            "spark.executor.instances": str(self.executor_count),
            "spark.executor.cores": str(self.executor_cores),
            "spark.executor.memory": f"{self.executor_memory_in_gb}G",
            "spark.executor.memoryOverhead": f"{self.yarn_memory_overhead_in_gb}G",
            "spark.memory.fraction": str(self.spark_memory_fraction),
            "spark.memory.stroageFraction": str(self.spark_memory_fraction),
        }

    def validate_cluster_id(self):
        """Ensure the cluster is running and has a valid id"""
        if not hasattr(self, "cluster_id"):
            raise EMRException(
                "Could not find a running cluster. Please call the start method"
            )

        if not self.validate_str(self.cluster_id):
            raise EMRException("Cluster id is invalid")


def _get_state(cluster_or_step: Any):
    return cluster_or_step["Status"]["State"]


def _get_reason(cluster_or_step: Any):
    """Get state change reason message."""
    # StateChangeReason is {} before the first state change
    return cluster_or_step["Status"]["StateChangeReason"].get("Message", "")


def _boto3_now():
    """Get a ``datetime`` that's compatible with :py:mod:`boto3`.
    These are always UTC time, with time zone ``dateutil.tz.tzutc()``.
    """
    if tzutc is None:
        raise ImportError("You must install dateutil to get boto3-compatible datetimes")

    return datetime.now(tzutc())


def strip_microseconds(delta):
    """Return the given :py:class:`datetime.timedelta`, without microseconds.
    Useful for printing :py:class:`datetime.timedelta` objects.
    """
    return timedelta(delta.days, delta.seconds)
