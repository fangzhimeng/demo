package com.sparkbyexamples.spark.delta

import org.apache.spark
import org.apache.spark.sql.{Dataset, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._
class deltaTest {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RDDToDataFrame").setMaster("local")
    val sc = new SparkContext(conf)
    val spark = new SQLContext(sc)
    spark.sql("set spark.sql.shuffle.partitions = 1")
    spark.sql("set spark.databricks.delta.snapshotPartitions = 1")
    val data:Dataset[Row] = spark.range(0, 5).toDF()
    import spark.implicits._
    data.write.format("delta").save("/tmp/delta-table")
    val df2 = spark.read.json(
      "src/main/resources/zipcodes_streaming/zipcode1.json",
      "src/main/resources/zipcodes_streaming/zipcode2.json")
    df2.show(false)
    df2.write
      .json("/tmp/spark_output/zipcodes.json")
  }

}
