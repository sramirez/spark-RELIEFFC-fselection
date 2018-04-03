package org.apache.spark.ml.util


import breeze.linalg._
import breeze.numerics._
import org.apache.spark.util.AccumulatorV2
import scala.collection.mutable.ArrayBuffer

/**
 * @author sramirez
 */
class COOAccumulator extends AccumulatorV2[ArrayBuffer[(Int, Int, Double)], ArrayBuffer[(Int, Int, Double)]] {
  
  def this(m: ArrayBuffer[(Int, Int, Double)]) = {
    this()
    this.accMatrix = m.clone()
  }

  private var accMatrix = new ArrayBuffer[(Int, Int, Double)]
  private var zero: Boolean = true

  def reset(): Unit = {
    accMatrix.clear()
    zero = true
  }

  def add(v: ArrayBuffer[(Int, Int, Double)]): Unit = {
    if(isZero) 
      zero = false
    accMatrix ++= v
  }
  
  def isZero(): Boolean = zero
  
  def merge(other: AccumulatorV2[ArrayBuffer[(Int, Int, Double)], ArrayBuffer[(Int, Int, Double)]]): Unit = add(other.value)
  
  def value: ArrayBuffer[(Int, Int, Double)] = accMatrix
  
  def copy(): AccumulatorV2[ArrayBuffer[(Int, Int, Double)], ArrayBuffer[(Int, Int, Double)]] = new COOAccumulator(accMatrix)
}