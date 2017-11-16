/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature

import scala.util.Random
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Matrix}
import org.apache.spark.ml.param.{IntParam, ParamValidators}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors
import scala.collection.mutable.Queue
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import scala.collection.immutable.Map
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.util.SizeEstimator
import org.apache.spark.ml.param.DoubleParam

/**
 * Params for [[LSH]].
 */
private[ml] trait LocalitySensitiveHashingParams extends HasInputCol with HasOutputCol {
  /**
   * Param for the number of hash tables used in LSH OR-amplification.
   *
   * LSH OR-amplification can be used to reduce the false negative rate. Higher values for this
   * param lead to a reduced false negative rate, at the expense of added computational complexity.
   * @group param
   */
  final val numHashTables: IntParam = new IntParam(this, "numHashTables", "number of hash " +
    "tables, where increasing number of hash tables lowers the false negative rate, and " +
    "decreasing it improves the running performance", ParamValidators.gt(0))

  /** @group getParam */
  final def getNumHashTables: Int = $(numHashTables)

  setDefault(numHashTables -> 1)
  
  /**
   * Param for the size of signatures built inside the hash tables.
   *
   * Higher values for this param lead to a reduced false negative rate, 
   * at the expense of added computational complexity.
   * @group param
   */
  final val signatureSize: IntParam = new IntParam(this, "signatureSize", "signature size or" +
    "number of random projections, where increasing size means lowers the false negative rate, and " +
    "decreasing it improves the running performance", ParamValidators.gt(0))

  /** @group getParam */
  final def getSignatureSize: Int = $(signatureSize)

  setDefault(signatureSize -> 16)
  
  /* if 0 we use sqrt(originalDim) */
  final val sparseSpeedup: DoubleParam = new DoubleParam(this, "sparseSpeedup", "", ParamValidators.gtEq(0))

  /** @group getParam */
  final def getSparseSpeedup: Double = $(sparseSpeedup)

  setDefault(sparseSpeedup -> 0)
  

  /**
   * Transform the Schema for LSH
   * @param schema The schema of the input dataset without [[outputCol]].
   * @return A derived schema with [[outputCol]] added.
   */
  protected[this] final def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(outputCol), DataTypes.createArrayType(new VectorUDT))
  }
}

/**
 * Model produced by [[LSH]].
 */
private[ml] abstract class LocalitySensitiveHashingModel[T <: LocalitySensitiveHashingModel[T]]
  extends Model[T] with LocalitySensitiveHashingParams with MLWritable {
  self: T =>
  /**
   * The hash function of LSH, mapping an input feature vector to multiple hash vectors.
   * @return The mapping of LSH function.
   */
  protected[ml] val hashFunction: Vector => Array[Vector]

  /**
   * Calculate the distance between two different keys using the distance metric corresponding
   * to the hashFunction.
   * @param x One input vector in the metric space.
   * @param y One input vector in the metric space.
   * @return The distance between x and y.
   */
  protected[ml] def keyDistance(x: Vector, y: Vector): Double

  /**
   * Calculate the distance between two different hash Vectors.
   *
   * @param x One of the hash vector.
   * @param y Another hash vector.
   * @return The distance between hash vectors x and y.
   */
  protected[ml] def hashDistance(x: Seq[Vector], y: Seq[Vector]): Double
  
    
  /**
   * Calculate the distance between two different hash Vectors, imposing a 
   * lower limit (+/- 1) to the distance, and a upper one to the number of projections 
   * that can differ (n).
   *
   * @param x One of the hash vector.
   * @param y Another hash vector.
   * @param n upper limit to the number of distinct projections.
   * @return The distance between hash vectors x and y (or positive infinity if invalid).
   */
  protected[ml] def hashThresholdedDistance(x: Seq[Vector], y: Seq[Vector], n: Int): Double = {
    // Since it's generated by hashing, it will be a pair of dense vectors.
    x.zip(y).map{ case(v1, v2) => 
      var dist = 0.0d
      var i = 0
      while(i < v1.size && dist < Double.PositiveInfinity){
        val offset = math.abs(v1(i) - v2(i))
        if(offset > 1 || dist > n) {
          dist = Double.PositiveInfinity
        } else {
          dist += math.pow(offset, 2)
        }
        i += 1
      }
      dist
    }.min
  }

  
  def transform(dataset: Dataset[_]): DataFrame

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
  
  // TODO: Fix the MultiProbe NN Search in SPARK-18454
  private[feature] def approxNearestNeighbors(
      dataset: Dataset[_],
      key: Vector,
      numNearestNeighbors: Int,
      probeMode: String,
      distCol: String,
      nelems: Long,
      relativeError: Double,
      step: Int = 2): Array[Row] = {
    require(numNearestNeighbors > 0, "The number of nearest neighbors cannot be less than 1")
    // Get Hash Value of the key
    
    val keyHash = hashFunction(key)
    val modelDataset: DataFrame = if (!dataset.columns.contains($(outputCol))) {
        transform(dataset)
      } else {
        dataset.toDF()
      }

    val modelSubset = probeMode match {
      
      case "single" => 
        def sameBucket(x: Seq[Vector], y: Seq[Vector]): Boolean = {
          x.zip(y).exists(tuple => tuple._1 == tuple._2)
        }
  
        // In the origin dataset, find the hash value that hash the same bucket with the key
        val sameBucketWithKeyUDF = udf((x: Seq[Vector]) =>
          sameBucket(x, keyHash), DataTypes.BooleanType)

        modelDataset.filter(sameBucketWithKeyUDF(col($(outputCol))))
      case "multi" =>
        // In the origin dataset, find the hash value that is closest to the key
        val distanceFunction = if(step < 1) (x: Seq[Vector]) => hashDistance(x, keyHash) else 
          (x: Seq[Vector]) => hashThresholdedDistance(x, keyHash, step)
        val hashDistUDF = udf(distanceFunction, DataTypes.DoubleType)
  
        // Compute threshold to get exact k elements.
        modelDataset.withColumn(distCol, hashDistUDF(col($(outputCol))))
            .filter(col(distCol) < Float.PositiveInfinity)
    }

    // Get the top k nearest neighbor by their distance to the key
    val keyDistUDF = udf((x: Vector) => keyDistance(x, key), DataTypes.DoubleType)
    val modelSubsetWithDistCol = modelSubset.withColumn(distCol, keyDistUDF(col($(inputCol))))
    implicit def rowOrder: Ordering[Row] = Ordering.by{_.getAs[Double](distCol)}
    val neighbors = modelSubsetWithDistCol.rdd.takeOrdered(numNearestNeighbors)(rowOrder)
    neighbors
  }

  private[feature] def approxNearestNeighbors(
      dataset: Dataset[_],
      key: Vector,
      numNearestNeighbors: Int,
      probeMode: String,
      distCol: String,
      nelems: Long): Array[Row] = {
    approxNearestNeighbors(dataset, key, numNearestNeighbors, probeMode, distCol, nelems, relativeError = 0.05)
  }

  /**
   * Given a large dataset and an item, approximately find at most k items which have the closest
   * distance to the item. If the [[outputCol]] is missing, the method will transform the data; if
   * the [[outputCol]] exists, it will use the [[outputCol]]. This allows caching of the
   * transformed data when necessary.
   *
   * @note This method is experimental and will likely change behavior in the next release.
   *
   * @param dataset The dataset to search for nearest neighbors of the key.
   * @param key Feature vector representing the item to search for.
   * @param numNearestNeighbors The maximum number of nearest neighbors.
   * @param distCol Output column for storing the distance between each result row and the key.
   * @return A dataset containing at most k items closest to the key. A column "distCol" is added
   *         to show the distance between each row and the key.
   */
  def approxNearestNeighbors(
      dataset: Dataset[_],
      key: Vector,
      numNearestNeighbors: Int,
      distCol: String,
      nelems: Long): Array[Row] = {
    approxNearestNeighbors(dataset, key, numNearestNeighbors, "multi", distCol, nelems)
  }

  /**
   * Overloaded method for approxNearestNeighbors. Use "distCol" as default distCol.
   */
  def approxNearestNeighbors(
      dataset: Dataset[_],
      key: Vector,
      numNearestNeighbors: Int,
      nelems: Long): Array[Row] = {
    approxNearestNeighbors(dataset, key, numNearestNeighbors, "multi", "distCol", nelems)
  }

  /**
   * Preprocess step for approximate similarity join. Transform and explode the [[outputCol]] to
   * two explodeCols: entry and value. "entry" is the index in hash vector, and "value" is the
   * value of corresponding value of the index in the vector.
   *
   * @param dataset The dataset to transform and explode.
   * @param explodeCols The alias for the exploded columns, must be a seq of two strings.
   * @return A dataset containing idCol, inputCol and explodeCols.
   */
  private[this] def processDataset(
      dataset: Dataset[_],
      inputName: String,
      explodeCols: Seq[String]): Dataset[_] = {
    require(explodeCols.size == 2, "explodeCols must be two strings.")
    val modelDataset: DataFrame = if (!dataset.columns.contains($(outputCol))) {
      transform(dataset)
    } else {
      dataset.toDF()
    }
    modelDataset.select(
      struct(col("*")).as(inputName), posexplode(col($(outputCol))).as(explodeCols))
  }

  /**
   * Recreate a column using the same column name but different attribute id. Used in approximate
   * similarity join.
   * @param dataset The dataset where a column need to recreate.
   * @param colName The name of the column to recreate.
   * @param tmpColName A temporary column name which does not conflict with existing columns.
   * @return
   */
  private[this] def recreateCol(
      dataset: Dataset[_],
      colName: String,
      tmpColName: String): Dataset[_] = {
    dataset
      .withColumnRenamed(colName, tmpColName)
      .withColumn(colName, col(tmpColName))
      .drop(tmpColName)
  }

  /**
   * Join two datasets to approximately find all pairs of rows whose distance are smaller than
   * the threshold. If the [[outputCol]] is missing, the method will transform the data; if the
   * [[outputCol]] exists, it will use the [[outputCol]]. This allows caching of the transformed
   * data when necessary.
   *
   * @param datasetA One of the datasets to join.
   * @param datasetB Another dataset to join.
   * @param threshold The threshold for the distance of row pairs.
   * @param distCol Output column for storing the distance between each pair of rows.
   * @return A joined dataset containing pairs of rows. The original rows are in columns
   *         "datasetA" and "datasetB", and a column "distCol" is added to show the distance
   *         between each pair.
   */
  def approxSimilarityJoin(
      datasetA: Dataset[_],
      datasetB: Dataset[_],
      threshold: Double,
      distCol: String): Dataset[_] = {

    val leftColName = "datasetA"
    val rightColName = "datasetB"
    val explodeCols = Seq("entry", "hashValue")
    val explodedA = processDataset(datasetA, leftColName, explodeCols)

    // If this is a self join, we need to recreate the inputCol of datasetB to avoid ambiguity.
    // TODO: Remove recreateCol logic once SPARK-17154 is resolved.
    val explodedB = if (datasetA != datasetB) {
      processDataset(datasetB, rightColName, explodeCols)
    } else {
      val recreatedB = recreateCol(datasetB, $(inputCol), s"${$(inputCol)}#${Random.nextString(5)}")
      processDataset(recreatedB, rightColName, explodeCols)
    }

    // Do a hash join on where the exploded hash values are equal.
    val joinedDataset = explodedA.join(explodedB, explodeCols)
      .drop(explodeCols: _*).distinct()

    // Add a new column to store the distance of the two rows.
    val distUDF = udf((x: Vector, y: Vector) => keyDistance(x, y), DataTypes.DoubleType)
    val joinedDatasetWithDist = joinedDataset.select(col("*"),
      distUDF(col(s"$leftColName.${$(inputCol)}"), col(s"$rightColName.${$(inputCol)}")).as(distCol)
    )

    // Filter the joined datasets where the distance are smaller than the threshold.
    joinedDatasetWithDist.filter(col(distCol) < threshold)
  }

  /**
   * Overloaded method for approxSimilarityJoin. Use "distCol" as default distCol.
   */
  def approxSimilarityJoin(
      datasetA: Dataset[_],
      datasetB: Dataset[_],
      threshold: Double): Dataset[_] = {
    approxSimilarityJoin(datasetA, datasetB, threshold, "distCol")
  }
  
}

/**
 * Locality Sensitive Hashing for different metrics space. Support basic transformation with a new
 * hash column, approximate nearest neighbor search with a dataset and a key, and approximate
 * similarity join of two datasets.
 *
 * This LSH class implements OR-amplification: more than 1 hash functions can be chosen, and each
 * input vector are hashed by all hash functions. Two input vectors are defined to be in the same
 * bucket as long as ANY one of the hash value matches.
 *
 * References:
 * (1) Gionis, Aristides, Piotr Indyk, and Rajeev Motwani. "Similarity search in high dimensions
 * via hashing." VLDB 7 Sep. 1999: 518-529.
 * (2) Wang, Jingdong et al. "Hashing for similarity search: A survey." arXiv preprint
 * arXiv:1408.2927 (2014).
 */
private[ml] abstract class LocalitySensitiveHashing[T <: LocalitySensitiveHashingModel[T]]
  extends Estimator[T] with LocalitySensitiveHashingParams with DefaultParamsWritable {
  self: Estimator[T] =>

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setNumHashTables(value: Int): this.type = set(numHashTables, value)
  
  /** @group setParam */
  def setSignatureSize(value: Int): this.type = set(signatureSize, value)
  
  /** @group setParam */
  def setSparseSpeedup(value: Double): this.type = set(sparseSpeedup, value)
  /**
   * Validate and create a new instance of concrete LSHModel. Because different LSHModel may have
   * different initial setting, developer needs to define how their LSHModel is created instead of
   * using reflection in this abstract class.
   * @param inputDim The dimension of the input dataset
   * @return A new LSHModel instance without any params
   */
  protected[this] def createRawLSHModel(projectedDim: Int, originalDim: Int, isSparse: Boolean): T

  override def fit(dataset: Dataset[_]): T = {
    transformSchema(dataset.schema, logging = true)
    val first = dataset.select(col($(inputCol))).head().get(0).asInstanceOf[Vector]
    val inputDim = first.size
    val isSparse = first.isInstanceOf[SparseVector]
    val model = createRawLSHModel($(signatureSize), inputDim, isSparse).setParent(this)
    copyValues(model)
  }
  
}

private[ml] trait LSHUtils {
  
  /**
   * Calculate the distance between two different keys using the distance metric corresponding
   * to the hashFunction.
   * @param x One input vector in the metric space.
   * @param y One input vector in the metric space.
   * @return The distance between x and y.
   */
  protected[ml] def keyDistance(x: Vector, y: Vector): Double

  /**
   * Calculate the distance between two different hash Vectors.
   *
   * @param x One of the hash vector.
   * @param y Another hash vector.
   * @return The distance between hash vectors x and y.
   */
  protected[ml] def hashDistance(x: Seq[Vector], y: Seq[Vector]): Double
  
  /**
   * The hash function of LSH, mapping an input feature vector to multiple hash vectors.
   * @return The mapping of LSH function.
   */
  protected[ml] val multipleHashFunction: (Vector, Broadcast[Array[Array[Vector]]], Double) => Array[Vector]
  
}
