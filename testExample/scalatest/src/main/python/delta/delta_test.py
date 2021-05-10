
from delta.tables import *
from pyspark.sql import SparkSession

spark2 = SparkSession.builder.appName("MyApp") .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0") .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog").getOrCreate()
data = spark2.range(0, 5)
data.write.format("delta").save("/tmp/delta-table")