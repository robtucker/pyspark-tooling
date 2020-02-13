# Grada pyspark utils

## Install

Pyspark utility functions


Install using the gemfury url containing your gemfury download token:

```
pip install pyspark-tooling
```


# EMR Deployment

This package contains certain class methods that are designed to allow you to get up and running with EMR very quickly, even if you are relatively new to EMR.

The simplest way to create a new EMR cluster and run your spark job is to use the `EMR.from_defaults()` static method.

```python
from pyspark_tooling.emr import EMR

emr = EMR.from_defaults(
    cluster_name="my_cluster"
    code_path="s3://my_bucket/main.py"
    log_uri="s3://my_bucket/logs"
    minimum_memory_in_gb=32,
    minimum_vcpu=12
)

emr.run()
```

After executing the `run` method, the `EMR` class will  create a new cluster and attempt to run the bootstrap script and the python files that you have provided. If it fails then the cluster will terminate (by default the cluster will terminate after job completion).

You can examine the exact configuration that was used to create your cluster by calling `emr.get_config()`.


### EMR calculations

The `EMR.from_defaults()` function uses the `InfrastructureConfig` class to calculate amongst other things:

- the optimal number of executors
- the amount of memory and vcpus given to each executor
- the default number of partitions
- the amount of memory reserved for yarn per executor


These calculations are made according to the [AWS best practice guide](https://aws.amazon.com/blogs/big-data/best-practices-for-successfully-managing-memory-for-apache-spark-applications-on-amazon-emr/).

There are 3 key parameters that you can pass to the `InfrastructureConfig` class constructor that will determine how your infrastructure is created.

1) `minimum_memory_in_gb` (default 32)

This paramter refers to the total memory that is available to your spark job across the entire cluster (note that some memory is also needed to run yarn, however this will be factored in automatically so you don't need to worry about it). Normally you should know in advance roughly how much memory you will need to hold all your data in memory. Note that every row in your RDD is saved in more than one partition for the sake of redundancy, which means the size of your RDD might be much larger the size of your original data. If you are doing wide transformations such as joins you will need even more memory to shuffle the data around. As the (official documentation)[https://spark.apache.org/docs/latest/hardware-provisioning.html] points out, the only way to know the size your RDD for sure is to examine the Storage tab in the Spark UI.

2) `vcpus_per_executor` (default 5)

This parameter refers to the number of vcpus that are assigned to each executor, however it has some important knock on effects. The number of executors is calculated by looking at how many vcpus are available across the entire cluster and dividing by the `vcpus_per_executor` parameter.

As an example, lets say there are 100 vcpus available across the entire cluster, and we use the default of 5 `vcpus_per_executor` as recommended by AWS. In this case we will end up with 100 / 5, which is 20 executors. 

Note that the amount of memory given to each executor follows from this calculation. Once we have calculated how many executors there will be, the available memory is evenly divided between all the executors. This parameter is therefore critical in determining 3 inter-related concerns, namely, how many vcpus each executor will have, thus how many executors there will be, and as a consequence, how much memory each of these executors will receive.

As an example, lets say you want to have less executors so that each executor has more memory, you can increase the `vcpus_per_executor` parameter up to 10. This means the same 100 vcpus will now be be divided by 10 (rather than 5), resulting in only 10 executors. Now each executor will get a tenth of the available memory.


3) `partitions_per_vcpu` (default 2) 

Each vcpu is given a certain number of partitions. This number is critical because it determines the amount of parallellism of your spark jobs. According to the best practice guide each vcpu can handle at least 2 partitions, sometimes more. However there may be scenarios where you actually want to limit the amount of parallellism in your cluster (this happens partiularly when you are doing a lot of wide transformations such as joins). In this case you could set `partitions_per_vcpu` to 1, so that each vcpu only has a single partition to deal with. This will reduce the number of partitions in your cluster. Note that each vcpu should have at the bare minimum 1 partition, although this could be as high as 4 or 5 if you want to get the most out of your cluster.


In summary, whilst these 3 parameters may seem simple, if you have understood everything above, then you should be able to tune your EMR cluster even for quite advanced use cases.


If these calculations are too restrictive for you and you already know exactly what you want from your cluster, then you should consider invoking the `EMR` class constructor directly. Please bear in mind that there are many paramaters that you will need to pass into the `EMR` class constructor, after making your own calculations. Please do not rely on any defaults as it is assumed you are making your own calculations.


# Transforms and Pipes

The motivation behind this repository is extremely opinionated. The functions in this repository are mostly *transforms*, which are designed to be used in *pipes*. 

A *transform* is a function which takes in a `DataFrame` and spits out another `DataFrame`. In other words it has the following signature:

```python
from pyspark.sql  import DataFrame

def identity_transform(df: DataFrame) -> DataFrame:
    """A transform that takes in a dataframe 
    and returns the same dataframe without any modifications"""
    return df
    
```

A pipe is a way of chaining transforms together into complex patterns.

As a simple example imagine we create a dataframe, then select a single column, then drop duplicates, and finally convert the dataframe to a list of tuples.

```python
from cytoolz import pipe
from functools import partial

from pyspark_tooling.dataframe import select_cols, deduplicate, to_tuples

data = [
    ('a', 1),
    ('a', 1),
    ('b', 2),
    ('c', 3),
    ('c', 4)
]

df = spark.createDataFrame(data, ['key', 'value'])

res = pipe(
    df,
    partial(select_cols, ['key']),
    dedeuplicate,
    to_tuples,
)


print(res) # outputs => [('a',), ('b',), ('c',)]

```

Note that the `select_cols` function takes 2 arguments, the first is a list of columns to be selected and the second is the dataframe from which to select them. All transforms *must* have `df` as an argument, however they may optionally have other other arguments as well. In this case, the dataframe is always the *final* argument. The other arguments may be "pre-filled" using the `partial` function. The resulting pipe is essentially a list of transforms, where each transform takes a single dataframe and spits out a single dataframe.


A valid question is why should I use transforms and pipes? This is primarily because of unit testing. Each individual transform does exactly one simple thing and as such can be unit tested separately. Chaining these transforms together will then produce complex transformations that nevertheless produce reliable results.


## Development

Before working on this repository, please ensure you have copied the `.env.example` file and renamed it as `.env` or otherwise ensure the environment variables listed in the `.env.example` file are available at runtime.

To run the test suite simply cd to the root directory and run `make test`. Please note that the test suite requires that docker is installed.

Please look at the `Makefile` for a list of available commands.
