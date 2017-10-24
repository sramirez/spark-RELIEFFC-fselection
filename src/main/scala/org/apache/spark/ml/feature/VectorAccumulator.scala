package org.apache.spark.ml.feature


import breeze.linalg._
import breeze.numerics._
import org.apache.spark.util.AccumulatorV2
/**
 * @author sramirez
 */
class VectorAccumulator(val rows: Int) extends AccumulatorV2[Vector[Float], Vector[Float]] {
  
  def this(m: Vector[Float]) = {
    this(m.size)
    this.accVector = m.copy
  }

  private var accVector: Vector[Float] = Vector.zeros[Float](rows)
  private var zero: Boolean = true

  def reset(): Unit = {
    accVector = Vector.zeros[Float](rows)
    zero = true
  }

  def add(v: Vector[Float]): Unit = {
    if(isZero) 
      zero = false
    accVector += v
  }
  
  def isZero(): Boolean = zero
  
  def merge(other: AccumulatorV2[Vector[Float], Vector[Float]]): Unit = accVector += other.value
  
  def value: Vector[Float] = accVector
  
  def copy(): AccumulatorV2[Vector[Float], Vector[Float]] = new VectorAccumulator(accVector)
}