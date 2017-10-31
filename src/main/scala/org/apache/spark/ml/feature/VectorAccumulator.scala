package org.apache.spark.ml.feature


import breeze.linalg._
import breeze.numerics._
import org.apache.spark.util.AccumulatorV2
/**
 * @author sramirez
 */
class VectorAccumulator(val rows: Int, sparse: Boolean) extends AccumulatorV2[Vector[Double], Vector[Double]] {
  
  def this(m: Vector[Double]) = {
    this(m.size, m.isInstanceOf[SparseVector[Double]])
    this.accVector = m.copy
  }

  private var accVector: Vector[Double] = if (sparse) SparseVector.zeros(rows) else Vector.zeros(rows)
  private var zero: Boolean = true

  def reset(): Unit = {
    accVector = if (sparse) SparseVector.zeros(rows) else Vector.zeros(rows)
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