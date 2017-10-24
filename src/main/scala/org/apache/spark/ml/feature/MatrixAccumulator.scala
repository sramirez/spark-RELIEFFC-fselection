package org.apache.spark.ml.feature


import breeze.linalg._
import breeze.numerics._
import org.apache.spark.util.AccumulatorV2
/**
 * @author sramirez
 */
class MatrixAccumulator(val rows: Int, val cols: Int) extends AccumulatorV2[Matrix[Double], Matrix[Double]] {
  
  def this(m: Matrix[Double]) = {
    this(m.rows, m.cols)
    this.accMatrix = m.copy
  }

  private var accMatrix: Matrix[Double] = Matrix.zeros[Double](rows, cols)
  private var zero: Boolean = true

  def reset(): Unit = {
    accMatrix = Matrix.zeros[Double](rows, cols)
    zero = true
  }

  def add(v: Matrix[Double]): Unit = {
    if(isZero) 
      zero = false
    accMatrix += v
  }
  
  def isZero(): Boolean = zero
  
  def merge(other: AccumulatorV2[Matrix[Double], Matrix[Double]]): Unit = accMatrix += other.value
  
  def value: Matrix[Double] = accMatrix
  
  def copy(): AccumulatorV2[Matrix[Double], Matrix[Double]] = new MatrixAccumulator(accMatrix)
}