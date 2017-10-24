package org.apache.spark.ml.feature


import breeze.linalg._
import breeze.numerics._
import org.apache.spark.util.AccumulatorV2
/**
 * @author sramirez
 */
class MatrixAccumulator(val rows: Int, val cols: Int) extends AccumulatorV2[Matrix[Float], Matrix[Float]] {
  
  def this(m: Matrix[Float]) = {
    this(m.rows, m.cols)
    this.accMatrix = m.copy
  }

  private var accMatrix: Matrix[Float] = Matrix.zeros[Float](rows, cols)
  private var zero: Boolean = true

  def reset(): Unit = {
    accMatrix = Matrix.zeros[Float](rows, cols)
    zero = true
  }

  def add(v: Matrix[Float]): Unit = {
    if(isZero) 
      zero = false
    accMatrix += v
  }
  
  def isZero(): Boolean = zero
  
  def merge(other: AccumulatorV2[Matrix[Float], Matrix[Float]]): Unit = accMatrix += other.value
  
  def value: Matrix[Float] = accMatrix
  
  def copy(): AccumulatorV2[Matrix[Float], Matrix[Float]] = new MatrixAccumulator(accMatrix)
}