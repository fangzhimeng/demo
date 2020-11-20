/**
  *
  * @ProjectName: testExample
  * @Package:
  * @ClassName: ClasstagTest
  * @description: ${description}
  * @Author: fangzhimeng
  * @CreateDate: 2020/11/20 14:24
  * @UpdateUser: 更新者
  * @UpdateDate: 2020/11/20 14:24
  * @UpdateRemark: 更新说明
  * @Version: 1.0
  * @create: 2020-11-20 14:24
  */
object ClasstagTest {


  def main(args: Array[String]): Unit = {
    class Animal
    val myMap: collection.Map[String, Any] = Map("Number" -> 1, "Greeting" -> "Hello World",
      "Animal" -> new Animal)
    /* 下面注释的代码将会不通过编译
     * Any不能被当时Int使用
     */
    //val number:Int = myMap("Number")
    //println("number is " + number)
    //使用类型转换，可以通过编译
    val number: Int = myMap("Number").asInstanceOf[Int]
    println("number  is " + number)
    //下面的代码将会抛出ClassCastException
    val greeting: String = myMap("Number").asInstanceOf[String]
  }
}
