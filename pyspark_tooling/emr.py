import os
import boto3
import json
import copy
import math
from enum import Enum
from typing import List


# AWS default iam roles
EMR_ROLE = "EMR_DefaultRole"
EMR_EC2_ROLE = "EMR_EC2_DefaultRole"


class ActionOnFailure(Enum):
    terminate_job_flow = "TERMINATE_JOB_FLOW"
    terminate_cluster = "TERMINATE_CLUSTER"
    cancel = "CANCEL_AND_WAIT"
    cont = "CONTINUE"


class MarketType(Enum):
    on_demand = "ON_DEMAND"
    spot = "SPOT"


class InstanceOptimization(Enum):
    general = "general"
    compute = "compute"
    memory = "memory"
    storage = "storage"


class InfrastructureConfig:

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
        minimum_memory_in_gb: int = 32,
        # optionally specify a minimum number of VCPUs
        minimum_vcpus: int = 4,
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
        self.minimum_memory_in_gb = minimum_memory_in_gb
        self.minimum_vcpus = minimum_vcpus
        self.yarn_memory_overhead_fraction = yarn_memory_overhead_fraction
        self.vcpus_per_executor = vcpus_per_executor
        self.partitions_per_vcpu = partitions_per_vcpu
        self.num_executors_for_driver = num_executors_for_driver

        # the total minimum memory includes the memory allocated to yarn
        self.total_minimum_memory = math.ceil(
            self.minimum_memory_in_gb
            + (self.minimum_memory_in_gb * self.yarn_memory_overhead_fraction)
        )

        # calculate which instance type should be used
        self.set_instance_type(instance_optimization)
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
        self.memory_per_instance = instances[index][0]
        self.vcpus_per_instance = instances[index][0]

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

    def vcpus_per_instance(self):
        """The number of vcpus that come with the instance less
        one vcpu, which is reserved for hadoop daemons"""
        return self.vcpus_per_instance - 1

    def executors_per_instance(self):
        """Based on a fixed number of vcpus per executor,
        how many executors can we fit into this instance (rounded down)?"""
        return math.floor(self.vcpus_per_instance() / self.vcpus_per_executor)

    def total_memory_per_executor(self):
        """Given how many executors we can fit into this instance,
        how much memory would each of these executors have?"""
        return int(self.memory_per_instance / self.executors_per_instance())

    def yarn_memory_per_executor(self):
        """How much of the total memory available to each
        executor in GB should be devoted to yarn"""
        return self.total_memory_per_executor() * self.yarn_memory_overhead_fraction

    def spark_memory_per_executor(self):
        """How much of the total memory available to each
        executor in GB should be devoted to spark"""
        return self.total_memory_per_executor() * (
            1 - self.yarn_memory_overhead_fraction
        )

    def total_executors(self):
        """The total number of executors across all instances of this type"""
        return self.executors_per_instance() * self.instance_count

    def total_vcpus(self):
        """The total number of vcpus for all instances
        minus the vcpus devoted to the hadoop daemon"""
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


    def arg_list(self):
        return {
            "executor_cores": config.vcpus_per_executor,
            "executor_memory_in_gb": conf.spark_memory_per_executor(),
            "yarn_memory_overhead_in_gb": conf.yarn_memory_per_executor(),
            "driver_cores": config.driver_vcpus(),
            "driver_memory_in_gb": config.driver_memory(),
            "executor_count": config.total_executors(),
            "default_parallellism": config.total_partitions(),
            "master_instance_type": config.instance_type,
            "master_instance_count": config.master_instance_count,
            "core_instance_type": config.instance_type,
            "core_instance_count": config.core_instance_count,
            "task_instance_type": None,
            "task_instance_count": 0,
        }


class EMR:
    def __init__(
        self,
        # name of the cluster as it appears in aws console
        cluster_name: str,
        # the path to the main entrypoint file
        code_entrypoint_path: str,
        # s3 path to the code zipped code bundle
        code_bundle_path: str,
        # s3 path to where the bootstrap script lives
        bootstrap_script_path: str,
        # s3 bucket where the output logs should be written
        logs_path: str,
        # a dictionary of environment variables that will be passed into the emr cluster
        emr_env_vars: dict,
        # the number of vcpus assigned to each executor
        executor_cores: int,
        # the memory given to each executor not including the fraction given to yarn
        executor_memory_in_gb: int,
        # the fraction of each executor's total memory devoted to yarn
        yarn_memory_overhead_in_gb: int,
        # the number of vcpu assigned to the driver
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
        # do not set this to true unless you know what you are doing
        maximise_resource_allocation: bool = False,
        # do not set this to true unless you know what you are doing
        dynamic_allocation_enabled: bool = False,
        # the amount of time to wait for the infrastructure to boot up
        infrastructure_timeout_duration_in_mins: int = 20,
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
        tags: dict = {},
        # what should emr do if one of the steps fails
        action_on_failure: ActionOnFailure = ActionOnFailure.terminate_cluster,
        # should this cluster only be visible to the user who started it
        visible_to_all_users: bool = True,
        # which version of emr should be used
        release_label: str = "emr-5.27.0",
        # which yarn scheduler should be used
        yarn_scheduler: str = "org.apache.hadoop.yarn.util.resource.DominantResourceCalculator",
        # the ec2 key for the spot instance
        ec2_key_name: str = None,
        # a list of subnet ids
        subnet_ids: str = None,
    ):
        self.cluster_name = cluster_name
        self.code_entrypoint_path = code_entrypoint_path
        self.code_bundle_path = code_bundle_path
        self.bootstrap_script_path = bootstrap_script_path
        self.logs_path = logs_path
        self.emr_env_vars = emr_env_vars
        self.executor_cores = executor_cores
        self.executor_memory_in_gb = executor_memory_in_gb
        self.yarn_memory_overhead_in_gb = yarn_memory_overhead_in_gb
        self.driver_cores = driver_cores
        self.driver_memory_in_gb = driver_memory_in_gb
        self.executor_count = executor_count
        self.default_parallellism = default_parallellism
        self.maximise_resource_allocation = maximise_resource_allocation
        self.dynamic_allocation_enabled = dynamic_allocation_enabled
        self.master_instance_type = master_instance_type
        self.master_instance_count = master_instance_count
        self.core_instance_type = core_instance_type
        self.core_instance_count = core_instance_count
        self.task_instance_type = task_instance_type
        self.task_instance_count = task_instance_count
        self.infrastructure_timeout_duration_in_mins = (
            infrastructure_timeout_duration_in_mins
        )
        self.market_type = market_type
        self.bid_price_as_percentage_of_on_demand = bid_price_as_percentage_of_on_demand
        self.job_flow_role = job_flow_role
        self.service_role = service_role
        self.applications = applications
        self.tags = tags
        self.action_on_failure = action_on_failure
        self.visible_to_all_users = visible_to_all_users
        self.release_label = release_label
        self.yarn_scheduler = yarn_scheduler
        self.maximise_resource_allocation = maximise_resource_allocation
        self.ec2_key_name = ec2_key_name
        self.subnet_ids = subnet_ids

    @staticmethod
    def from_defaults(
        cluster_name: str,
        code_path: str,
        bootstrap_script_path: str,
        emr_env_vars: dict,
        minimum_memory_in_gb: int = 32,
        minimum_vcpus: int = 4,
        partitions_per_vcpu: int = 2,
        vcpus_per_executor: int = 5,
        partitions_per_vcpu: int = 2,
        yarn_memory_overhead_fraction: int = 0.1,
        instance_optimization: InstanceOptimization = InstanceOptimization.memory,
        num_executors_for_driver: int = 1,
        market_type: MarketType = MarketType.spot,
        bid_price_as_percentage_of_on_demand: int = 80,
        job_flow_role: str = EMR_EC2_ROLE,
        service_role: str = EMR_ROLE,
        tags: dict = {},
        ec2_key_name: str = None,
        subnet_ids: str = None,
    ):

        config = InfrastructureConfig(
            minimum_memory_in_gb=minimum_memory_in_gb,
            minimum_vcpus=minimum_vcpus,
            vcpus_per_executor=vcpus_per_executor,
            partitions_per_vcpu=partitions_per_vcpu,
            yarn_memory_overhead_fraction=yarn_memory_overhead_fraction,
            instance_optimization=instance_optimization,
            num_executors_for_driver=num_executors_for_driver,
        )

        return EMR(
            cluster_name=cluster_name,
            code_path=code_path,
            bootstrap_script_path=bootstrap_script_path,
            emr_env_vars=emr_env_vars,
            maximise_resource_allocation=False,
            dynamic_allocation_enabled=False,
            instance_optimization=instance_optimization,
            num_executors_for_driver=num_executors_for_driver,
            market_type=market_type,
            bid_price_as_percentage_of_on_demand=bid_price_as_percentage_of_on_demand,
            job_flow_role=job_flow_role,
            service_role=service_role,
            tags=tags,
            ec2_key_name=ec2_key_name
            subnet_ids=subnet_ids,
            # add the infrastructure config
            **config.arg_list(),
        )

    def run(self, synchronous=False):
        if synchronous:
            return self.run_sync()
        return self.run_async()

    def run_sync(self):
        pass

    def run_async(self):
        """Run the EMR cluster asynchronously"""
        client = boto3.client("emr")

        client.run_job_flow(
            Name=self.cluster_name,
            LogUri=self.log_uri,
            ReleaseLabel=self.release_label,
            Instances=self.get_infrastructure_config(),
            Steps=self.get_spark_steps(),
            Configurations=self.get_emr_config(),
            Applications=self.applications,
            BootstrapActions=self.get_spark_bootstrap_actions(),
            VisibleToAllUsers=self.visible_to_all_users,
            JobFlowRole=self.job_flow_role,
            ServiceRole=self.service_role,
            Tags=self.tags,
        )
        return {"statusCode": 200, "body": json.dumps("Cluster Launched")}

    def get_job_command(self):
        """The command that will be run as the main step"""
        return f""

    def get_env_vars(self):
        required_env_vars = {"PYSPARK_DEFAULT_PARALLELLISM": self.default_parallellism}
        # merge the required env vars with the ones supplied by the user
        return copy.deepcopy(required_env_vars).update(self.emr_env_vars)

    def get_spark_steps(self):
        return [
            {
                "Name": "Setup Debugging",
                "ActionOnFailure": self.action_on_failure,
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": ["state-pusher-script"],
                },
            },
            {
                "Name": "Run ifx job",
                "ActionOnFailure": self.action_on_failure,
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": self.get_job_command().split(),
                },
            },
        ]

    def get_spark_bootstrap_actions(self):
        return [
            {
                "Name": "Bootstrap script",
                "ScriptBootstrapAction": {"Path": self.bootstrap_script_path},
            }
        ]

    def get_infrastructure_config(self):
        """Choose the relevant instance config"""
        if self.market_type == MarketType.spot:
            return self.get_spot_config()
        return self.get_on_demand_config()

    def get_on_demand_config(self):
        """The on demand config is the simplest infrastructure config to use"""
        return {
            "KeepJobFlowAliveWhenNoSteps": False,
            "InstanceGroups": [
                {
                    "Name": "Master",
                    "InstanceCount": self.master_instance_count,
                    "InstanceType": self.master_instance_type,
                    "Market": "ON_DEMAND",
                    "InstanceRole": "MASTER",
                },
                {
                    "Name": "Core",
                    "InstanceCount": self.core_instance_count,
                    "InstanceType": self.core_instance_type,
                    "Market": "ON_DEMAND",
                    "InstanceRole": "CORE",
                },
            ],
        }

    def get_spot_config(self):
        """The spot config should reduce the cost of running emr quite drastically.
        Set the bid price as a percentage of demand price accordingly"""
        return {
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
                },
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
                },
            ],
            "Ec2SubnetIds": self.subnet_ids,
            "Ec2KeyName": self.ec2_key_name,
            "KeepJobFlowAliveWhenNoSteps": False,
        }

    def get_spark_defaults(self):
        """Configuration values used in tuning spark jobs"""
        return {
            "spark.dynamicAllocation.enabled": str(
                self.dynamic_allocation_enabled
            ).lower(),
            "spark.executor.cores": str(self.executor_cores),
            "spark.executor.memory": f"{self.executor_memory_in_gb}G",
            "spark.yarn.executor.memoryOverhead": f"{self.yarn_memory_overhead_in_gb}G",
            "spark.driver.memory": f"{self.driver_memory_in_gb}G",
            "spark.driver.cores": str(self.driver_cores),
            "spark.executor.instances": str(self.executor_count),
            "spark.default.parallelism": str(self.default_parallellism),
        }

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
