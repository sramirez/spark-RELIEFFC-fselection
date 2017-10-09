package org.apache.spark.ml.feature


import breeze.linalg._
import breeze.numerics._
import org.apache.spark.util.AccumulatorV2
/**
 * @author sramirez
 */
class VectorAccumulator(val rows: Int) extends AccumulatorV2[Vector[Double], Vector[Double]] {
  
  def this(m: Vector[Double]) = {
    this(m.size)
    this.accVector = m.copy
  }

  private var accVector: Vector[Double] = Vector.zeros[Double](rows)
  private var zero: Boolean = true

  def reset(): Unit = {
    accVector = Vector.zeros[Double](rows)
    zero = true
  }

  def add(v: Vector[Double]): Unit = {
    if(isZero) 
      zero = false
    accVector += v
  }
  
  def isZero(): Boolean = zero
  
  def merge(other: AccumulatorV2[Vector[Double], Vector[Double]]): Unit = accVector += other.value
  
  def value: Vector[Double] = accVector
  
  def copy(): AccumulatorV2[Vector[Double], Vector[Double]] = new VectorAccumulator(accVector)
}