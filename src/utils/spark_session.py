from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import os
from pathlib import Path

def get_spark_session(app_name: str, master: str, shuffle_partitions: int, timezone: str):
    # Required for local Spark + Delta on Windows
    os.environ["HADOOP_HOME"] = r"C:\hadoop" # update with your path
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "") # update with your path
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", timezone)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.HDFSLogStore")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.warehouse.dir", str(Path("data") / "_spark_warehouse"))
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark