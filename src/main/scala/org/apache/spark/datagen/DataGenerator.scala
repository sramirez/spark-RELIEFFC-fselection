package org.apache.spark.datagen

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext

/**
 * @author sramirez
 */
object DataGenerator {
  
    var sqlContext: SQLContext = null
  var pathFile = "test_lung_s3.csv"
  var order = -1 // Note: -1 means descending
  var nPartitions = 1
  var nTop = 10
  var discretize = false
  var padded = 2
  var classLastIndex = false
  var clsLabel: String = null
  var inputLabel: String = "features"
  var firstHeader: Boolean = false
  var k: Int = 5
  var categorical: Boolean = false
  var nselect: Int = 10
  var seed = 12345678L
  
  
    def main(args: Array[String]) {
      val initStartTime = System.nanoTime()
      
      val conf = new SparkConf().setAppName("CollisionFS Test").setMaster("local[*]")
      val sc = new SparkContext(conf)
      sqlContext = new SQLContext(sc)
  
      println("Usage: MLlibTest --train-file=\"hdfs://blabla\" --nselect=10 --npart=1 --categorical=false --k=5 --ntop=10 --disc=false --padded=2 --class-last=true --header=false")
          
      // Create a table of parameters (parsing)
      val params = args.map{ arg =>
          val param = arg.split("--|=").filter(_.size > 0)
          param.size match {
            case 2 =>  param(0) -> param(1)
            case _ =>  "" -> ""
          }
      }.toMap    
      
      pathFile = params.getOrElse("train-file", "src/test/resources/data/test_lung_s3.csv")
      nPartitions = params.getOrElse("npart", "1").toInt
      nTop = params.getOrElse("ntop", "10").toInt
      discretize = params.getOrElse("disc", "false").toBoolean
      padded = params.getOrElse("padded", "2").toInt
      classLastIndex = params.getOrElse("class-last", "false").toBoolean
      firstHeader = params.getOrElse("header", "false").toBoolean
      k = params.getOrElse("k", "10").toInt
      nselect = params.getOrElse("nselect", "10").toInt
      categorical = params.getOrElse("categorical", "false").toBoolean
      
      //DiscreteDataGenerator.generate(sc, nRelevantFeatures = 10, nDataPoints = 200000, 
      //    noiseOnRelevant = 0.1, redundantNoises = (0), nRandomFeatures, 
      //    outputPath, nLabels, maxBins, maxDepth, seed)
      println("Params used: " +  params.mkString("\n"))
    }
}