package com.fzm.flink

/**
  *
  * @ProjectName: testExample
  * @Package: com.fzm.flink
  * @ClassName: StreamingJob
  * @description: ${description}
  * @Author: fangzhimeng
  * @CreateDate: 2020/11/24 10:59
  * @UpdateUser: 更新者
  * @UpdateDate: 2020/11/24 10:59
  * @UpdateRemark: 更新说明
  * @Version: 1.0
  * @create: 2020-11-24 10:59
  */
import org.apache.flink.streaming.api.scala._
import org.apache.flink.table.sources._
import org.apache.flink.table.api.scala.StreamTableEnvironment
import org.apache.flink.table.api._
import org.apache.flink.types.Row
import org.apache.flink.table.api.{
  TableEnvironment,
  TableSchema,
  Types,
  ValidationException
}
import org.apache.flink.api.java.io.jdbc.JDBCAppendTableSink
import org.apache.flink.api.common.typeinfo.TypeInformation

object StreamingJob {
  def main(args: Array[String]) {
    val SourceCsvPath =
      "d://dictionary.csv"
    val CkJdbcUrl =
      "jdbc:clickhouse://192.168.1.43:8123/default"
    val CkUsername = "default"
    val CkPassword = "123456"
    val BatchSize = 500 // 设置您的batch size

    val env = StreamExecutionEnvironment.getExecutionEnvironment

   // val tEnv = StreamTableEnvironment.create(env)
   val bsSettings = EnvironmentSettings.newInstance().useBlinkPlanner().inStreamingMode().build()
    val tEnv = StreamTableEnvironment.create(env, bsSettings)

    val csvTableSource = CsvTableSource
      .builder()
      .path(SourceCsvPath)
      .ignoreFirstLine()
      .fieldDelimiter(",")
      .field("name", Types.STRING)
      .field("age", Types.STRING)
      .field("sex", Types.STRING)
      .field("grade", Types.STRING)
      .field("rate", Types.STRING)
      .field("rate2", Types.STRING)
      .field("rate3", Types.STRING)
      .field("rate4", Types.STRING)
      .build()

    tEnv.registerTableSource("source", csvTableSource)

    val resultTable = tEnv.scan("source").select("name, grade, rate")

    val insertIntoCkSql =
      """
        |  INSERT INTO sink_table (
        |    name, grade, rate
        |  ) VALUES (
        |    ?, ?, ?
        |  )
      """.stripMargin

    //将数据写入 ClickHouse Sink
    val sink = JDBCAppendTableSink
      .builder()
      .setDrivername("ru.yandex.clickhouse.ClickHouseDriver")
      .setDBUrl(CkJdbcUrl)
      .setUsername(CkUsername)
      .setPassword(CkPassword)
      .setQuery(insertIntoCkSql)
      .setBatchSize(BatchSize)
      .setParameterTypes(Types.STRING, Types.STRING, Types.STRING)
      .build()

    tEnv.registerTableSink(
      "sink",
      Array("name", "grade", "rate"),
      Array(Types.STRING, Types.STRING, Types.STRING),
      sink
    )

    tEnv.insertInto(resultTable, "sink")

    env.execute("Flink Table API to ClickHouse Example")
  }
}
//clickhouse-client  -u default --password 123456
//CREATE TABLE sink_table( id UInt16,name String,grade String,rate String ,create_date date ) ENGINE = MergeTree(create_date, (id), 8192);
