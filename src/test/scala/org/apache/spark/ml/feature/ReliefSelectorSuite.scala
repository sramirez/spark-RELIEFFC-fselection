package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.junit.runner.RunWith
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.scalatest.junit.JUnitRunner
import TestHelper._


/**
  * Test information theoretic feature selection on datasets from Peng's webpage
  *
  * @author Sergio Ramirez
  */
@RunWith(classOf[JUnitRunner])
class ReliefSelectorSuite extends FunSuite with BeforeAndAfterAll {

  var sqlContext: SQLContext = null

  override def beforeAll(): Unit = {
    sqlContext = new SQLContext(SPARK_CTX)
  }
  
  /** Do mRMR feature selection on LUNG data. */
  test("Run RELIEF-F on lung data (nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_lung_s3.csv")
    val cols = df.columns
    val pad = 2
    val allVectorsDense = true
    val discreteData = true
    
    val model = getSelectorModel(sqlContext, df, cols.drop(1), cols.head, 
        10, discreteData, allVectorsDense, pad)

    assertResult("29, 223, 10, 19, 172, 55, 183, 23, 35, 56") {
      model.getSelectedFeatures.mkString(", ")
    }
  }
  
  /** Do mRMR feature selection on COLON data. */
  test("Run RELIEF-F on colon data (nfeat = 10)") {

    val df = readCSVData(sqlContext, "test_colon_s3.csv")
    val cols = df.columns
    val pad = 2
    val allVectorsDense = true
    val discreteData = true
    val model = getSelectorModel(sqlContext, df, cols.drop(1), cols.head, 
        10, discreteData, allVectorsDense, pad)

    assertResult("1422, 248, 74, 244, 266, 764, 1413, 1771, 1152, 779") {
      model.getSelectedFeatures.mkString(", ")
    }
  }
}