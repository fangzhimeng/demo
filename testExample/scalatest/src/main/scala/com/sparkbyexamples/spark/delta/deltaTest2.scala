package com.sparkbyexamples.spark.delta

import org.apache.spark
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Dataset, SQLContext}

object deltaTest2 {

  val conf = new SparkConf().setAppName("RDDToDataFrame").setMaster("local")
  val sc = new SparkContext(conf)
  val spark = new SQLContext(sc)

}
