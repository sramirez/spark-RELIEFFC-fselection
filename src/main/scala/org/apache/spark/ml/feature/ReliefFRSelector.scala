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

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.{ Since, Experimental }
import org.apache.spark.ml._
import org.apache.spark.ml.attribute.{ AttributeGroup, _ }
import org.apache.spark.ml.linalg.{ Vector, VectorUDT }
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.linalg.{ Vectors => OldVectors }
import org.apache.spark.mllib.regression.{ LabeledPoint => OldLabeledPoint }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ DoubleType, StructField, StructType }
import scala.collection.mutable.HashMap
import org.apache.spark.broadcast.Broadcast
import scala.collection.mutable.Queue
import breeze.linalg.DenseMatrix
import breeze.linalg.Axis
import breeze.linalg.DenseVector
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.types.DataTypes
import scala.collection.mutable.WrappedArray
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.FloatType
import breeze.linalg.VectorBuilder
import breeze.linalg.CSCMatrix
import org.apache.spark.ml.linalg.SparseVector
import breeze.linalg.mapActiveValues

/**
 * Params for [[ReliefFRSelector]] and [[ReliefFRSelectorModel]].
 */
private[feature] trait ReliefFRSelectorParams extends Params
    with HasInputCol with HasOutputCol with HasLabelCol with HasSeed {

  /**
   * Relief with redundancy removal criterion used to rank the features.
   * Relief relies on LSH to perform efficient nearest neighbor searches.
   *
   * @group param
   */
  
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
  
  /**
   * The length of each hash bucket, a larger bucket lowers the false negative rate. The number of
   * buckets will be `(max L2 norm of input vectors) / bucketLength`.
   *
   *
   * If input vectors are normalized, 1-10 times of pow(numRecords, -1/inputDim) would be a
   * reasonable value
   * @group param
   */
  val bucketLength: DoubleParam = new DoubleParam(this, "bucketLength",
    "the length of each hash bucket, a larger bucket lowers the false negative rate.",
    ParamValidators.gt(0))

  /** @group getParam */
  final def getBucketLength: Double = $(bucketLength)  
  setDefault(bucketLength -> 4)

  /**
   * Number of features that selector will select (ordered by statistic value descending). If the
   * number of features is < numTopFeatures, then this will select all features. The default value
   * of numTopFeatures is 50.
   *
   * @group param
   */
  final val numTopFeatures = new IntParam(this, "numTopFeatures",
    "Number of features that selector will select, ordered by statistics value descending. If the" +
      " number of features is < numTopFeatures, then this will select all features.",
    ParamValidators.gtEq(1))
  setDefault(numTopFeatures -> 25)
  
  final val numNeighbors = new IntParam(this, "numNeighbors", "",
    ParamValidators.gtEq(1))
  setDefault(numNeighbors -> 10)
  
  final val estimationRatio: DoubleParam = new DoubleParam(this, "estimationRatio", "", ParamValidators.inRange(0,1))
  setDefault(estimationRatio -> 0.25)
  
  final val batchSize: DoubleParam = new DoubleParam(this, "batchSize", "", ParamValidators.inRange(0,1))
  setDefault(batchSize -> 0.1)
  
  final val queryStep: IntParam = new IntParam(this, "queryStep", "", ParamValidators.gtEq(1))
  setDefault(queryStep -> 2)  
  
  final val lowerFeatureThreshold: DoubleParam = new DoubleParam(this, "lowerFeatureThreshold", "", ParamValidators.inRange(0,1))
  setDefault(lowerFeatureThreshold -> 0.5)
  
  final val redundancyRemoval: BooleanParam = new BooleanParam(this, "redundancyRemoval", "")
  setDefault(redundancyRemoval -> true)

}

/**
 * :: Experimental ::
 * Relief feature selection, which relies on distance measurements among neighbors to weight features.
 */
@Experimental
final class ReliefFRSelector @Since("1.6.0") (@Since("1.6.0") override val uid: String = Identifiable.randomUID("ReliefFRSelector"))
    extends Estimator[ReliefFRSelectorModel] with ReliefFRSelectorParams with DefaultParamsWritable {

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setSeed(value: Long): this.type = set(seed, value)
  def setNumHashTables(value: Int): this.type = set(numHashTables, value)
  def setSignatureSize(value: Int): this.type = set(signatureSize, value)  
  def setBucketLength(value: Double): this.type = set(bucketLength, value)
  def setNumTopFeatures(value: Int): this.type = set(numTopFeatures, value)
  def setNumNeighbors(value: Int): this.type = set(numNeighbors, value)
  def setEstimationRatio(value: Double): this.type = set(estimationRatio, value)
  def setBatchSize(value: Double): this.type = set(batchSize, value)
  def setQueryStep(value: Int): this.type = set(queryStep, value) 
  def setLowerFeatureThreshold(value: Double): this.type = set(lowerFeatureThreshold, value)
  def setRedundancyRemoval(value: Boolean): this.type = set(redundancyRemoval, value)
  
  // Case class for criteria/feature
  case class F(feat: Int, crit: Double)
  
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
  
  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): ReliefFRSelectorModel = {
    
    transformSchema(dataset.schema, logging = true)
    
    // Get Hash Value of the key 
    val hashOutputCol = "hashRELIEF_" + System.currentTimeMillis()
    val brp = new BucketedRandomProjectionLSH()
      .setNumHashTables($(numHashTables))
      .setInputCol($(inputCol))
      .setOutputCol(hashOutputCol)
      .setBucketLength($(bucketLength))
      .setSignatureSize($(signatureSize))
      .setSeed($(seed))
    val LSHmodel = brp.fit(dataset)
    
    val struct = dataset.schema.fields(dataset.schema.fieldIndex($(inputCol)))
    val continuous = AttributeGroup.fromStructField(struct).attributes.get.exists { att => att.isNumeric }
    val modelDataset: DataFrame = if (!dataset.columns.contains(hashOutputCol)) {
        LSHmodel.transform(dataset)
      } else {
        dataset.toDF()
      }
    val weights = Array.fill((1 / $(batchSize)).toInt)($(estimationRatio) * $(batchSize))
    modelDataset.show()
    val batches = modelDataset.randomSplit(weights, $(seed))
    
    // Get some basic information about the dataset
    val sc = modelDataset.sparkSession.sparkContext
    val spark = modelDataset.sparkSession.sqlContext
    val nFeat = modelDataset.select($(inputCol)).head().getAs[Vector](0).size
    val nElems = modelDataset.count()
    val priorClass = modelDataset.select($(labelCol)).rdd.map{ case Row(label: Double) => label }
        .countByValue()
        .mapValues(v => (v.toDouble / nElems).toFloat)
        .map(identity).toMap
        
    val schema = new StructType()
            .add(StructField("id", IntegerType, true))
            .add(StructField("score", FloatType, true))
    var featureWeights: DenseVector[Float] = DenseVector.zeros(nFeat)
    var jointMatrix: DenseMatrix[Double] = DenseMatrix.zeros(nFeat, nFeat)
    var marginalVector: DenseVector[Double] = DenseVector.zeros(nFeat)
    var topFeatures: Map[Int, Float] = (0 until nFeat).map(i => i -> 0.0f).toMap
    
    var total = 0L // total number of comparisons at the collision level
    for(batchIndex <- 0 until batches.size) {
      val query = batches(batchIndex)
      val modelQuery: DataFrame = if (!query.columns.contains(hashOutputCol)) {
        LSHmodel.transform(query)
      } else {
        query.toDF()
      }  
    
      // Index query objects and compute the table that indicates where are located its neighbors
      val idxModelQuery = modelQuery.withColumn("UniqueID", monotonically_increasing_id)
      val bFullQuery: Broadcast[Array[Row]] = sc.broadcast(idxModelQuery.select(
          col("UniqueID"), col($(inputCol)), col($(labelCol)), col(hashOutputCol)).collect())
          
      val neighbors: RDD[(Long, Map[Short, Iterable[Int]])] = approxNNByPartition(modelDataset, 
          bFullQuery, LSHmodel, $(numNeighbors) * priorClass.size)
      
      val bNeighborsTable: Broadcast[Map[Long, Map[Short, Iterable[Int]]]] = 
          sc.broadcast(neighbors.collectAsMap().toMap)
      
      val (rawWeights: RDD[(Int, Float)], partialJoint, partialMarginal, partialCount, skipped) = computeReliefWeights(
          modelDataset.select($(inputCol), $(labelCol)), bFullQuery, bNeighborsTable, 
          priorClass, nFeat, nElems, continuous, topFeatures)

      val partialRelief = rawWeights.cache()
      partialRelief.collect().par.foreach{ case(k,v) => featureWeights(k) += v } 
      total += partialCount.value
      jointMatrix += partialJoint.value
      marginalVector += partialMarginal.value
      println("# omitted instances in this step: " + skipped)
      
      // Normalize previous results and return the best features
      val partialReliefDF = spark.createDataFrame(partialRelief.map{ case(k,v) => Row(k,v)}, schema)
      val normalized = normalizeRankingDF(partialReliefDF).cache()
      normalized.show
      val quantile = normalized.stat.approxQuantile("score", Array($(lowerFeatureThreshold)), 0.05)(0)
      topFeatures = normalized.filter(normalized.col("score").geq(quantile)).collect().map { 
        case Row(id: Int, score: Float) => id -> score}.toMap
    }
        
    val maxRelief = breeze.linalg.max(featureWeights)
    val minRelief = breeze.linalg.min(featureWeights)
    val finalWeights = featureWeights.mapActiveValues(score => 
        (score - minRelief) / (maxRelief - minRelief))
    // normalized redundancy
    val redundancyMatrix = computeRedudancy(jointMatrix, marginalVector, total, nFeat, continuous)
    val (reliefCol, relief) = selectFeatures(finalWeights, redundancyMatrix)
    val outRC = reliefCol.map { case F(feat, rel) => (feat + 1) + "\t" + "%.4f".format(rel) }.mkString("\n")
    val outR = relief.map { case F(feat, rel) => (feat + 1) + "\t" + "%.4f".format(rel) }.mkString("\n")
    println("\n*** RELIEF + Collisions selected features ***\nFeature\tScore\n" + outRC)
    println("\n*** RELIEF selected features ***\nFeature\tScore\n" + outR)
    
    val model = new ReliefFRSelectorModel(uid, relief.map(_.feat).toArray, reliefCol.map(_.feat).toArray)
    copyValues(model)
  }
  
  private def normalizeRankingDF(partialWeights: Dataset[_]) = {
      val scores = partialWeights.agg(min("score"), max("score")).head()
      val maxRelief = scores.getAs[Float](1); val minRelief = scores.getAs[Float](0)
      val normalizeUDF = udf((x: Float) => (x - minRelief) / (maxRelief - minRelief), DataTypes.FloatType)
      partialWeights.withColumn("score", normalizeUDF(col("score")))
  }
  
  private def computeReliefWeights (
      modelDataset: Dataset[_],
      bModelQuery: Broadcast[Array[Row]],
      bNeighborsTable: Broadcast[Map[Long, Map[Short, Iterable[Int]]]],
      priorClass: Map[Double, Float],
      nFeat: Int,
      nElems: Long,
      isCont: Boolean,
      topFeatures: Map[Int, Float]
      ) = {
    
     // Initialize accumulators for RELIEF+Collision computation
    val sc = modelDataset.sparkSession.sparkContext
    val accMarginal = new VectorAccumulator(nFeat); sc.register(accMarginal, "marginal")
    val accJoint = new MatrixAccumulator(nFeat, nFeat);  sc.register(accJoint, "joint")
    val totalInteractions = sc.longAccumulator("totalInteractions")
    val omittedInstances = sc.longAccumulator("omittedInstances")
    val label2Num = priorClass.zipWithIndex.map{case (k, v) => k._1 -> v.toShort}
    val idxPriorClass = priorClass.zipWithIndex.map{case (k, v) => v -> k._2}
    val bTF = sc.broadcast(topFeatures)
    val lowerDistanceTh = .8f
    val lowerFeatureTh = $(lowerFeatureThreshold)    
    val sparse = true
    
    val rawReliefWeights = modelDataset.rdd.mapPartitionsWithIndex { 
      case(pindex, it) =>
        val localExamples = it.toArray
        val query: Array[Row] = bModelQuery.value
        val table: Map[Long, Map[Short, Iterable[Int]]] = bNeighborsTable.value
        // last position is reserved to negative weights from central instances.
        val marginal = breeze.linalg.DenseVector.zeros[Double](nFeat)
        
        val joint = breeze.linalg.DenseMatrix.zeros[Double](nFeat, nFeat)
        val reliefWeights = breeze.linalg.DenseVector.zeros[Float](nFeat)         
        val r = new scala.util.Random($(seed))
        // Data are assumed to be scaled to have 0 mean, and 1 std
        val vote = if(isCont) (d: Double) => 1 - math.min(6.0, d) / 6.0 else (d: Double) => Double.MinPositiveValue
        
        query.map{ case Row(qid: Long, qinput: Vector, qlabel: Double, _) =>
            
            val rnumber = r.nextFloat()
            val condition = if(isCont) 6 * (1 - (lowerDistanceTh + rnumber * lowerDistanceTh)) else 0.0f
            val featThreshold = lowerFeatureTh + rnumber * lowerFeatureTh
            val condition2 = bTF.value.mapValues{ _ >= featThreshold}
            val localRelief = breeze.linalg.DenseMatrix.zeros[Double](nFeat, label2Num.size)
            val classCounter = breeze.linalg.DenseVector.zeros[Long](label2Num.size)
        
            table.get(qid) match { case Some(localMap) =>
                val neighbors = localMap.getOrElse(pindex.toShort, Iterable.empty)
                val boundary = neighbors.exists { i => val Row(_, nlabel: Double) = localExamples(i)
                  nlabel != qlabel }
                if(boundary) { // Only boundary points allowed, omitted those homogeneous with the class
                  neighbors.map{ lidx =>
                    val Row(ninput: Vector, nlabel: Double) = localExamples(lidx)
                    var mainCollisioned = Queue[Int](); var auxCollisioned = Queue[Int]() // annotate similar features
                    val pcounter = Array.fill(nFeat)(0.0d) // isolate the strength of collision by feature
                    val jointVote = if(isCont) (i1: Int, i2: Int) => (pcounter(i1) + pcounter(i2)) / 2 else 
                      (i1: Int, i2: Int) => pcounter(i1)
                    val labelIndex = label2Num.get(nlabel.toFloat).get
                    classCounter(labelIndex) += 1
                          
                    qinput.foreachActive{ (index, value) =>
                       val fdistance = math.abs(value - ninput(index))
                       //// RELIEF Computations
                       localRelief(index, labelIndex) += fdistance.toFloat
                       //// Check if there exist a collision
                       // The closer the distance, the more probable.
                       if(fdistance <= condition){
                          val contribution = vote(fdistance)
                          marginal(index) += contribution
                          pcounter(index) = contribution
                          val fit = mainCollisioned.iterator
                          while(fit.hasNext){
                            val i2 = fit.next
                            joint(i2, index) += jointVote(index, i2)
                          } 
                          // Check if redundancy is relevant here. Depends on the feature' score in the previous stage.
                          if(condition2.getOrElse(index, false)){ // Relevant, the feature is added to the main group and update auxiliar feat's.
                            mainCollisioned += index
                            val fit = auxCollisioned.iterator
                            while(fit.hasNext){
                              val i2 = fit.next
                              joint(i2, index) += jointVote(index, i2)
                            }
                          } else { // Irrelevant, added to the secondary group
                            auxCollisioned += index
                          }
                       }
                    }
                  }  
                } else {
                  omittedInstances.add(1)
                }
              case None =>
                System.err.println("Instance does not found in the table")
            }
         val denom = 1 - priorClass.get(qlabel).get
         val indWeights = localRelief.mapActivePairs{ case((feat, cls), value) =>  
           if(cls != qlabel){
             ((idxPriorClass.get(cls).get / denom) * value / classCounter(cls)).toFloat 
           } else {
             (-value / classCounter(cls)).toFloat
           }
         }
         reliefWeights += breeze.linalg.sum(indWeights, Axis._1)
         totalInteractions.add(classCounter.sum)
      }
      // update accumulated matrices  
      accMarginal.add(marginal)
      accJoint.add(joint)
      reliefWeights.activeIterator
    }.reduceByKey(_ + _)
     
    (rawReliefWeights, accJoint, accMarginal, totalInteractions, omittedInstances)
  }
  
  private def computeReliefWeightsSparse (
      modelDataset: Dataset[_],
      bModelQuery: Broadcast[Array[Row]],
      bNeighborsTable: Broadcast[Map[Long, Map[Short, Iterable[Int]]]],
      priorClass: Map[Double, Float],
      nFeat: Int,
      nElems: Long,
      isCont: Boolean,
      topFeatures: Map[Int, Float]
      ) = {
    
     // Initialize accumulators for RELIEF+Collision computation
    val sc = modelDataset.sparkSession.sparkContext
    val accMarginal = new VectorAccumulator(nFeat); sc.register(accMarginal, "marginal")
    val accJoint = new MatrixAccumulator(nFeat, nFeat);  sc.register(accJoint, "joint")
    val totalInteractions = sc.longAccumulator("totalInteractions")
    val omittedInstances = sc.longAccumulator("omittedInstances")
    val label2Num = priorClass.zipWithIndex.map{case (k, v) => k._1 -> v.toShort}
    val idxPriorClass = priorClass.zipWithIndex.map{case (k, v) => v -> k._2}
    val bTF = sc.broadcast(topFeatures)
    val lowerDistanceTh = .8f
    val lowerFeatureTh = $(lowerFeatureThreshold)    
    val sparse = true
    
    val rawReliefWeights = modelDataset.rdd.mapPartitionsWithIndex { 
      case(pindex, it) =>
        val localExamples = it.toArray
        val query: Array[Row] = bModelQuery.value
        val table: Map[Long, Map[Short, Iterable[Int]]] = bNeighborsTable.value
        // last position is reserved to negative weights from central instances.
        val marginal = new VectorBuilder[Int](nFeat)
        val joint = new CSCMatrix.Builder[Int](rows=nFeat, cols=nFeat)
        val reliefWeights = breeze.linalg.DenseVector.zeros[Float](nFeat)         
        val r = new scala.util.Random($(seed))
        
        query.map{ case Row(qid: Long, qinput: SparseVector, qlabel: Double, _) =>
            
            val rnumber = r.nextFloat()
            val condition = if(isCont) 6 * (1 - (lowerDistanceTh + rnumber * lowerDistanceTh)) else 0.0f
            val featThreshold = lowerFeatureTh + rnumber * lowerFeatureTh
            val condition2 = bTF.value.mapValues{ _ >= featThreshold}
            val localRelief = new CSCMatrix.Builder[Float](rows=nFeat, cols=label2Num.size)
            val classCounter = breeze.linalg.DenseVector.zeros[Long](label2Num.size)
        
            table.get(qid) match { case Some(localMap) =>
                val neighbors = localMap.getOrElse(pindex.toShort, Iterable.empty)
                val boundary = neighbors.exists { i => val Row(_, nlabel: Double) = localExamples(i)
                  nlabel != qlabel }
                if(boundary) { // Only boundary points allowed, omitted those homogeneous with the class
                  neighbors.map{ lidx =>
                    val Row(ninput: SparseVector, nlabel: Double) = localExamples(lidx)
                    var mainCollisioned = Queue[Int](); var auxCollisioned = Queue[Int]() // annotate similar features
                    val labelIndex = label2Num.get(nlabel.toFloat).get
                    classCounter(labelIndex) += 1
                    
                    val v1Indices = qinput.indices; val nnzv1 = v1Indices.length; var kv1 = 0
                    val v2Indices = ninput.indices; val nnzv2 = v2Indices.length; var kv2 = 0
                    while (kv1 < nnzv1 || kv2 < nnzv2) {
                      var score = 0.0
                      var index = kv1
            
                      if (kv2 >= nnzv2 || (kv1 < nnzv1 && v1Indices(kv1) < v2Indices(kv2))) {
                        score = qinput.values(kv1); kv1 += 1; index = kv1
                      } else if (kv1 >= nnzv1 || (kv2 < nnzv2 && v2Indices(kv2) < v1Indices(kv1))) {
                        score = ninput.values(kv2); kv2 += 1; index = kv2
                      } else {
                        score = qinput.values(kv1) - ninput.values(kv2); kv1 += 1; kv2 += 1; index = kv1
                      }
                      val fdistance = math.abs(score)
                      //// RELIEF Computations
                      localRelief.add(index, labelIndex, fdistance.toFloat)
                      //// Check if there exist a collision
                      // The closer the distance, the more probable.
                      if(fdistance <= condition){
                        marginal.add(index, 1)
                        val fit = mainCollisioned.iterator
                        while(fit.hasNext){
                          val i2 = fit.next
                          joint.add(i2, index, 1)
                        } 
                        // Check if redundancy is relevant here. Depends on the feature' score in the previous stage.
                        if(condition2.getOrElse(index, false)){ // Relevant, the feature is added to the main group and update auxiliar feat's.
                          mainCollisioned += index
                          val fit = auxCollisioned.iterator
                          while(fit.hasNext){
                            val i2 = fit.next
                            joint.add(i2, index, 1)
                          }
                        } else { // Irrelevant, added to the secondary group
                          auxCollisioned += index
                        }
                       }
                    }
                  }  
                } else {
                  omittedInstances.add(1)
                }
              case None =>
                System.err.println("Instance does not found in the table")
            }
         val denom = 1 - priorClass.get(qlabel).get
         val indWeights = localRelief.result.mapActivePairs{ case((feat, cls), value) =>  
           if(cls != qlabel){
             ((idxPriorClass.get(cls).get / denom) * value / classCounter(cls)).toFloat 
           } else {
             (-value / classCounter(cls)).toFloat
           }
         }
         reliefWeights += breeze.linalg.sum(indWeights, Axis._1)
         totalInteractions.add(classCounter.sum)
      }(breeze.linalg.support.CanMapKeyValuePairs[breeze.linalg.CSCMatrix[Float],(Int, Int),Float,Float,Float])
      // update accumulated matrices  
      accMarginal.add(marginal.toSparseVector)
      accJoint.add(joint.result)
      reliefWeights.activeIterator
    }.reduceByKey(_ + _)
     
    (rawReliefWeights, accJoint, accMarginal, totalInteractions, omittedInstances)
  }
  
  private def computeRedudancy(rawJoint: DenseMatrix[Double], rawMarginal: DenseVector[Double], 
      total: Long, nFeat: Int, isCont: Boolean) = {
    // Now compute redundancy based on collisions and normalize it
    val factor = if(isCont) 1.0 else Double.MinPositiveValue
    val marginal = rawMarginal.mapPairs{ case(_, e) => e /  (total * factor) }  
    val joint = rawJoint.mapPairs{ case(_, e) => e /  (total * factor) }  
    
    // Compute mutual information using collisions with and without class
    val redundancyMatrix = breeze.linalg.DenseMatrix.zeros[Float](nFeat, nFeat)
    var maxRed = Float.NegativeInfinity
    var minRed = Float.PositiveInfinity
    joint.activeIterator.foreach { case((i1,i2), value) =>
      if(i1 < i2) {
        val red = (value * log2(value / (marginal(i1) * marginal(i2)))).toFloat  
        val correctedRed = if(!red.isNaN()) red else 0
        redundancyMatrix(i1, i2) = correctedRed; redundancyMatrix(i2, i1) = correctedRed 
        if(correctedRed > maxRed) {
          maxRed = correctedRed 
        } else if(correctedRed < minRed) {
          minRed = correctedRed
        }
      }        
    }
    redundancyMatrix.mapActiveValues { e => ((e - minRed) / (maxRed - minRed)).toFloat }
  }
  
  // TODO: Fix the MultiProbe NN Search in SPARK-18454
  private def approxNNByPartition(
      modelDataset: Dataset[_],
      bModelQuery: Broadcast[Array[Row]],
      model: LSHModel[_],
      k: Int): RDD[(Long, Map[Short, Iterable[Int]])] = {
    
    case class Localization(part: Short, index: Int)
    val sc = modelDataset.sparkSession.sparkContext
    val hashDistance = model.hashThresholdedDistance(_: Seq[Vector], _: Seq[Vector], $(queryStep))
    val realDistance = model.keyDistance(_: Vector, _: Vector)
    
    val neighbors = modelDataset.select($(inputCol), model.getOutputCol).rdd.mapPartitionsWithIndex { 
        case (pindex, it) => 
          // Initialize the map composed by the priority queue and the central element's ID
          val query = bModelQuery.value // Fields: "UniqueID", $(inputCol), $(labelCol), "hashOutputCol"
          val ordering = Ordering[Float].on[(Float, Localization)](-_._1)   
          val neighbors = query.map { case Row(id: Long, _, _, _) => 
            id -> new BoundedPriorityQueue[(Float, Localization)](k)(ordering)
          }   
      
          var i = 0
          // First iterate over the local elements, and then over the sampled set (also called query set).
          while(it.hasNext) {
            val Row(inputNeig: Vector, hashNeig: WrappedArray[Vector]) = it.next
            (0 until query.size).foreach { j => 
               val Row(_, inputQuery: Vector, _, hashQuery: WrappedArray[Vector]) = query(j) 
               val hdist = hashDistance(hashQuery.array, hashNeig.array)
               if(hdist < Double.PositiveInfinity) {
                 val distance = realDistance(inputQuery, inputNeig)
                 neighbors(j)._2 += distance.toFloat -> Localization(pindex.toShort, i)
               }
                 
            }
            i += 1              
          }            
          neighbors.toIterator
      }.reduceByKey(_ ++= _).mapValues(
          _.map(l => Localization.unapply(l._2).get).groupBy(_._1).mapValues(_.map(_._2)).map(identity)) 
      // map(identity) needed to fix bug: https://issues.scala-lang.org/browse/SI-7005
    neighbors
  }
  
   def selectFeatures(reliefRanking: DenseVector[Float],
      redundancyMatrix: breeze.linalg.DenseMatrix[Float]): (Seq[F], Seq[F]) = {
    
     val attAlive = Array.fill(reliefRanking.size)(true)
    // Initialize all (except the class) criteria with the relevance values
    val pool = reliefRanking.map(mi => new FeatureScore().init(mi.toFloat)).toArray.zipWithIndex.par
    
    // Get the maximum and initialize the set of selected features with it
    val (max, mid) = pool.maxBy(_._1.relevance)
    var selected = Seq(F(mid, max.score))
    max.valid = false
    
    var moreFeat = true
    // Iterative process for redundancy and conditional redundancy
    while (selected.size < $(numTopFeatures) && moreFeat) {

      attAlive(selected.head.feat) = false
      val redundancies = redundancyMatrix(::, selected.head.feat)
              .toArray
              .zipWithIndex
              .filter(c => attAlive(c._2))

      // Update criteria with the new redundancy values      
      redundancies.foreach({
        case (mi, k) =>            
          pool(k)._1.update(mi.toFloat)
      })
      
      // select the best feature and remove from the whole set of features
      val validFeat = pool.filter(_._1.valid)
      if(!validFeat.isEmpty){
        val (max, maxi) = validFeat.maxBy(c => (c._1.score, -c._2))      
        selected = F(maxi, max.score) +: selected
        max.valid = false
      } else {
        moreFeat = false
      }
    }
    val reliefNoColl = reliefRanking.toArray.zipWithIndex.map{ case(score, id) => F(id, score)}
        .sortBy(f => (-f.crit, f.feat))
        .slice(0, $(numTopFeatures)).toSeq
    (selected.reverse, reliefNoColl)  
  }
  
  private class FeatureScore {
    var relevance: Float = 0.0f
    var redundance: Float = 0.0f
    var selectedSize: Int = 0
    var valid = true
  
    def score = {
      if (selectedSize > 0) {
        relevance - redundance / selectedSize
      } else {
        relevance
      }
    }
    
    def init(relevance: Float): FeatureScore = {
      this.relevance = relevance
      this
    }
    
    def update(mi: Float): FeatureScore = {
      redundance += mi
      selectedSize += 1
      this
    }
  } 
   
  private def log2(x: Double) = { math.log(x) / math.log(2) }

  override def copy(extra: ParamMap): ReliefFRSelector = defaultCopy(extra)
}

@Since("1.6.0")
object ReliefFRSelector extends DefaultParamsReadable[ReliefFRSelector] {

  @Since("1.6.0")
  override def load(path: String): ReliefFRSelector = super.load(path)
}

/**
 * :: Experimental ::
 * Model fitted by [[ReliefFRSelector]].
 */
@Experimental
final class ReliefFRSelectorModel private[ml] (
  @Since("1.6.0") override val uid: String,
  private val reliefFeatures: Array[Int],
  private val reliefColFeatures: Array[Int]
)
    extends Model[ReliefFRSelectorModel] with ReliefFRSelectorParams with MLWritable {

  import ReliefFRSelectorModel._
  
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setRedundancyRemoval(value: Boolean): this.type = set(redundancyRemoval, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformedSchema = transformSchema(dataset.schema, logging = true)
    val newField = transformedSchema.last

    // TODO: Make the transformer natively in ml framework to avoid extra conversion.
    val selectedFeatures = if($(redundancyRemoval)) reliefFeatures else reliefColFeatures
    val transformer: Vector => Vector = v =>  FeatureSelectionUtils.compress(OldVectors.fromML(v), selectedFeatures).asML
    val selector = udf(transformer)

    dataset.withColumn($(outputCol), selector(col($(inputCol))), newField.metadata)
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    val newField = prepOutputField(schema)
    val outputFields = schema.fields :+ newField
    StructType(outputFields)
  }

  /**
   * Prepare the output column field, including per-feature metadata.
   */
  private def prepOutputField(schema: StructType): StructField = {
    val origAttrGroup = AttributeGroup.fromStructField(schema($(inputCol)))
    val featureAttributes: Array[Attribute] = if (origAttrGroup.attributes.nonEmpty) {
      origAttrGroup.attributes.get.zipWithIndex.filter(x => reliefFeatures.contains(x._2)).map(_._1)
    } else {
      Array.fill[Attribute](reliefFeatures.size)(NumericAttribute.defaultAttr)
    }
    val newAttributeGroup = new AttributeGroup($(outputCol), featureAttributes)
    newAttributeGroup.toStructField()
  }

  override def copy(extra: ParamMap): ReliefFRSelectorModel = {
    val copied = new ReliefFRSelectorModel(uid, reliefFeatures, reliefColFeatures)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new ReliefFRSelectorModelWriter(this)
}

@Since("1.6.0")
object ReliefFRSelectorModel extends MLReadable[ReliefFRSelectorModel] {

  private[ReliefFRSelectorModel] class ReliefFRSelectorModelWriter(instance: ReliefFRSelectorModel) extends MLWriter {

    private case class Data(reliefFeatures: Array[Int], reliefColFeatures: Array[Int])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.reliefFeatures, instance.reliefColFeatures)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class ReliefFRSelectorModelReader extends MLReader[ReliefFRSelectorModel] {

    private val className = classOf[ReliefFRSelectorModel].getName

    override def load(path: String): ReliefFRSelectorModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      val Row(reliefFeatures: Array[Int], reliefColFeatures: Array[Int]) =
        MLUtils.convertVectorColumnsToML(data, "reliefFeatures", "reliefColFeatures")
          .select("originalMin", "originalMax")
          .head()
      val model = new ReliefFRSelectorModel(metadata.uid, reliefFeatures, reliefColFeatures)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  @Since("1.6.0")
  override def read: MLReader[ReliefFRSelectorModel] = new ReliefFRSelectorModelReader

  @Since("1.6.0")
  override def load(path: String): ReliefFRSelectorModel = super.load(path)
}
