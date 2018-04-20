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
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.MatrixAccumulator
import org.apache.spark.ml.util.VectorAccumulator
import org.apache.spark.ml.util.{BoundedPriorityQueue => BPQ}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.linalg.{ Vectors => OldVectors }
import org.apache.spark.mllib.regression.{ LabeledPoint => OldLabeledPoint }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ DoubleType, StructField, StructType }
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.FloatType
import org.apache.spark.util.SizeEstimator


import scala.collection.mutable.WrappedArray
import scala.collection.mutable.Queue
import scala.collection.mutable.HashMap
import scala.collection.immutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.collection.immutable.TreeMap

import breeze.linalg.Axis
import breeze.linalg.VectorBuilder
import breeze.linalg.CSCMatrix
import breeze.linalg.{Matrix => BM, Vector => BV, DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV}
import breeze.generic.UFunc
import breeze.generic.MappingUFunc
import breeze.linalg.support._

/**
 * Params for [[ReliefFRSelector]] and [[ReliefFRSelectorModel]].
 */
private[feature] trait ReliefFRSelectorParams extends Params
    with HasInputCol with HasOutputCol with HasLabelCol with HasSeed {

  /**
   * Relief + redundancy removal based on co-occurrences. Both are used to rank features.
   *
   * @group param
   */
    
  /**
   * Number of features that selector will select (ordered by statistic value descending). If the
   * number of features is < numTopFeatures, then this will select all features. The default value
   * of numTopFeatures is 50.
   *
   * @group param
   */
  final val numTopFeatures = new IntParam(this, "numTopFeatures",
    "Number of features that selector will select, ordered by statistics value descending. If the" +
      " number of features is < numTopFeatures, then the entire set of input features will be selected.",
    ParamValidators.gtEq(1))
  setDefault(numTopFeatures -> 10)
  
  /**
   * Number of neighbors used in RELIEF-F to estimate separability among features.
   * A more crowded neighborhood implies more accurate estimations, but a more costly process.
   * Note: The algorithm will select N-neighbors for each class so the real number of neighbors computed
   * will depend on the amount of output labels. The default value of numNeighbors is 10.
   *
   * @group param
   */
  final val numNeighbors = new IntParam(this, "numNeighbors", 
      "Number of neighbors used in RELIEF-F to estimate separability among features.",
      ParamValidators.gtEq(1))
  setDefault(numNeighbors -> 10)
  
  /**
   * Subset of samples that will serve to estimate weights. Notice that neighborhoods will be computed
   * regarding the entire dataset. Complexity order will thus shift from quadratic to one based on
   * estimationRate *  |dataset|. The default value of estimationRatio is 0.25.
   * 
   * 
   * @group param
   */
  final val estimationRatio: DoubleParam = new DoubleParam(this, "estimationRatio", "", ParamValidators.inRange(0,1))
  setDefault(estimationRatio -> 0.25)
  
  /**
   * Previous sample is split into several batches to reduce even further derived complexity. In addition,
   * ranking from previous steps will serve to select which features will be involved in redundancy computations.
   * The default value of estimationRatio is 0.25.
   * 
   * @group param
   */
  final val batchSize: DoubleParam = new DoubleParam(this, "batchSize", "", ParamValidators.inRange(0,1))
  setDefault(batchSize -> 0.25)
  
  /**
   * Rate of features (with respect to the selection set) that will be involved in redundancy computations. 
   * Low values close to 1.0 imply quicker computations but weaker estimations, 
   * whereas high values imply the opposite scenario. 
   * The default value of lowerFeatureThreshold is 3.0 (three times the selection rate).
   *
   * @group param
   */
  final val lowerFeatureThreshold: DoubleParam = new DoubleParam(this, "lowerFeatureThreshold", "", ParamValidators.gtEq(1))
  setDefault(lowerFeatureThreshold -> 3.0)
  
  /** 
   *  For continuous attributes, this parameter defines the percentage of maximum distance accounted in REDUNDACY estimations.
   *  Instead of defining the largest distance as the difference between the maximum and the minimun value in the dataset,
   *  we rely on Chebyshev's inequality to define the range as that formed by 6 standard deviations. This removes the effect of
   *  highly skewed outliers in data. Chebyshev's inequality states a minimum of 89% values are within three standard deviations.
   *  IMPORTANT: As prerequisite, data must be normalized to have 0 mean, and 1 standard deviation.  
   */
  final val lowerDistanceThreshold: DoubleParam = new DoubleParam(this, "lowerDistanceThreshold", 
      "For continuous attributes, this parameter defines the percentage of maximum distance accounted in estimations.", ParamValidators.inRange(0,1))
  setDefault(lowerDistanceThreshold -> 0.8)
  
  /**
   * If redundancy removal technique is activated or only standard RELIEF-F is performed. 
   * Redundancy estimation is based on zero-distances computed in the RELIEF-F step. 
   * If two features share similar values (distance = 0), redundancy score will increase.
   * The default value of redundancyRemoval is false.
   *
   * @group param
   */
  final val redundancyRemoval: BooleanParam = new BooleanParam(this, "redundancyRemoval", 
      "If redundancy removal technique is activated or only standard RELIEF-F is performed.")
  setDefault(redundancyRemoval -> false)
  
  /**
   * If input data are discrete, or continuous. Continuous data must have 0 mean, and 1 std for better performing in REDUNDANCY estimations. 
   * See lowerDistanceThreshold variable for more information.
   */
  final val discreteData: BooleanParam = new BooleanParam(this, "discreteData", 
      "Continuous data must have 0 mean, and 1 std for better performing in REDUNDANCY estimations.")
  setDefault(discreteData -> false)
  
}

/**
 * :: Experimental ::
 * Relief feature weighting method. It relies on distance measurements in neighborhoods to estimate feature importance.
 */
@Experimental
final class ReliefFRSelector @Since("1.6.0") (@Since("1.6.0") override val uid: String = Identifiable.randomUID("ReliefFRSelector"))
    extends Estimator[ReliefFRSelectorModel] with ReliefFRSelectorParams with DefaultParamsWritable {

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setSeed(value: Long): this.type = set(seed, value)
  def setDiscreteData(value: Boolean): this.type = set(discreteData, value)
  /** @group RELIEF params */
  def setNumTopFeatures(value: Int): this.type = set(numTopFeatures, value)
  def setNumNeighbors(value: Int): this.type = set(numNeighbors, value)
  def setEstimationRatio(value: Double): this.type = set(estimationRatio, value)
  def setBatchSize(value: Double): this.type = set(batchSize, value)
  def setLowerFeatureThreshold(value: Double): this.type = set(lowerFeatureThreshold, value)
  def setLowerDistanceThreshold(value: Double): this.type = set(lowerDistanceThreshold, value)
  def setRedundancyRemoval(value: Boolean): this.type = set(redundancyRemoval, value)
  
  
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
  
  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): ReliefFRSelectorModel = {
    
    transformSchema(dataset.schema, logging = true)
    
    // Get Hash Value of the key 
    val modelDataset: DataFrame = dataset.toDF()
    
    // Get some basic information about the dataset
    val sc = dataset.sparkSession.sparkContext
    val spark = dataset.sparkSession.sqlContext
    val nElems = modelDataset.count() // needed to persist the training set
    val first = modelDataset.head().getAs[Vector]($(inputCol))
    val sparse = first.isInstanceOf[SparseVector] // data is sparse or dense
    val nFeat = first.size
    val lowerFeat = math.max($(numTopFeatures), 
        math.round($(lowerFeatureThreshold).toFloat * $(numTopFeatures)))
        // Top-N relevant features used to estimate redundancy (default: 3 times the number of features selected)
    // Prior prob. for each output label
    val priorClass = modelDataset.select($(labelCol)).rdd.map{ case Row(label: Double) => label }
        .countByValue()
        .mapValues(v => (v.toDouble / nElems).toFloat)
        .map(identity).toMap
    
    // Sample estimation rate
    val sample = modelDataset.sample(false, $(estimationRatio), $(seed))
    val sampledSize = sample.count()
    val maxSizeAllowed = if(!sparse) Integer.MAX_VALUE / 8 / (nFeat + 2) / sampledSize else $(batchSize) 
    // In order to avoid batches broadcasted exceed the maximum set to Integer.MAX_VALUE (about 2,3 GB)
    val maxBatchSize = math.min($(batchSize), maxSizeAllowed)
    val nbatches = (1 / maxBatchSize).toInt
    
    // Display basic information
    logInfo("# of elements: " + nElems)
    logInfo("# of features: " + nFeat)        
    logInfo("Sparse data: " + sparse)
    logInfo("Computed lower threshold for features: " + lowerFeat)
    logInfo("Class distribution: " + priorClass.toString())
    logInfo("Sampled size: " + sampledSize)
    logInfo("Sampling percentage per batch: " + maxBatchSize)
    logInfo("Number of batches: " + nbatches)
    
    // Split into batches
    val weights = Array.fill(nbatches)(maxBatchSize)
    val batches = sample.randomSplit(weights, $(seed))
    var topFeatures: Set[Int] = Set.empty
    var featureWeights: BV[Float] = if (sparse) BSV.zeros(nFeat) else BV.zeros(nFeat)
    var marginalVector: BV[Double] = if (sparse) BSV.zeros(nFeat) else BV.zeros(nFeat)
    var total = 0L // total number of comparisons already made at the redundancy level
    val results: Array[RDD[(Int, (Float, Vector))]] = Array.fill(batches.size)(sc.emptyRDD)
        
    for(i <- 0 until batches.size) {
      val start = System.currentTimeMillis
      // Index query objects and compute the table that indicates where each neighbor is located
      val idxModelQuery: Dataset[Row] = batches(i).withColumn("UniqueID", monotonically_increasing_id).cache()
      val query: Dataset[Row] = idxModelQuery.select(col("UniqueID"), col($(inputCol)), col($(labelCol)))
      val lquery: Array[Row] = query.collect()
      
      logInfo("# of query elements: " + lquery.length)
      logInfo("Estimated size for broadcasted query: " + SizeEstimator.estimate(lquery)) 
      
      // Fully broadcast the queried elements (the entire batch)
      val bFullQuery: Broadcast[Array[Row]] = sc.broadcast(lquery)
      // For each query element, return N neighbors for each output label.
      val neighbors: RDD[(Long, Map[Int, Iterable[Int]])] = approxNNByPartition(idxModelQuery, 
          bFullQuery, $(numNeighbors) * priorClass.size)
      // Neighbors' positions are again broadcasted as a table (partition, neighbors' positions)
      val bNeighborsTable: Broadcast[Map[Long, Map[Int, Iterable[Int]]]] = 
          sc.broadcast(neighbors.collectAsMap().toMap)
      
      // Weights obtained in the RELIEF-F phase. Each structure is composed by: feature ID, relevance, redundancy.  
      val (rawWeights: RDD[(Int, (Float, Vector))], partialMarginal, partialCount) =  
          if (!sparse) computeReliefWeights(
              idxModelQuery.select($(inputCol), $(labelCol)), bFullQuery, bNeighborsTable, 
          topFeatures, priorClass, nFeat, nElems)
          else  computeReliefWeightsSparse(
              idxModelQuery.select($(inputCol), $(labelCol)), bFullQuery, bNeighborsTable, 
          topFeatures, priorClass, nFeat, nElems)
      
      // Normalize previous results and return the most relevant features for this batch/step
      results(i) = rawWeights.cache()      
      val localR = results(i).collect()
      if(results(i).count > 0){ // call the action required to persist data
        val normalized = normalizeRankingDF(results(i).mapValues(_._1))
        implicit def scoreOrder: Ordering[(Int, Float)] = Ordering.by{ _._2 }
        topFeatures = normalized.takeOrdered(lowerFeat)(scoreOrder.reverse).map(_._1).toSet
      }
        
      // Partial statistics for redundancy
      total += partialCount.value     
      marginalVector = marginalVector match {
        case sv: BSV[Double] => sv += partialMarginal.value.asInstanceOf[BSV[Double]]
        case dv: BDV[Double] => dv += partialMarginal.value
      }
        
      // Free some resources
      bNeighborsTable.destroy(); bFullQuery.destroy(); idxModelQuery.unpersist();
      val btime = (System.currentTimeMillis - start) / 1000
      logInfo("Batch #" + i + " computed in " + btime + "s")
    }
    
    // Once all batches have been computed, we join and aggregate results
    var tmpWeights: RDD[(Int, (Float, Vector))] = results(0)
    (1 until results.size).foreach{ i => tmpWeights = tmpWeights.union(results(i)) }
    val finalWeights = tmpWeights.reduceByKey({ case((r1, j1), (r2, j2)) => 
      (r1 + r2, Vectors.fromBreeze(j1.asBreeze + j2.asBreeze)) }).cache()
    val nWeights = finalWeights.count()

    // Unpersist partial results
    (0 until batches.size).foreach(i => results(i).unpersist())
    
    // min-max normalize final RELIEF-F weights
    val onlyWeights = finalWeights.values.map(_._1)
    val maxRelief = onlyWeights.max(); val minRelief = onlyWeights.min()
    val normWeights = finalWeights.mapValues{ case(w, joint) => (w - minRelief) / (maxRelief - minRelief) -> joint}
        
    // Compute and normalize redundancy score, and show feature ranking with and w/o the redundancy factor
    val rddFinalWeights = computeRedudancy(normWeights, marginalVector, total, nFeat, maxBatchSize, sparse).cache()
    
    val (reliefRed, stdRelief) = selectFeatures(rddFinalWeights, nFeat)
    val outRC = reliefRed.map { case F(feat, score) => (feat + 1) + "\t" + score.toString() }.mkString("\n")
    val outR = stdRelief.map { case F(feat, score) => (feat + 1) + "\t" + score.toString() }.mkString("\n")
    println("\n*** RELIEF + Collisions selected features ***\nFeature\tScore\tRelevance\tRedundancy\n" + outRC)
    println("\n*** RELIEF selected features ***\nFeature\tScore\tRelevance\tRedundancy\n" + outR)
    
    val model = new ReliefFRSelectorModel(uid, stdRelief.map(_.feat).toArray, reliefRed.map(_.feat).toArray)
    copyValues(model)
  }
  
  /** Normalize ranking in each iteration to know which features should be involved in redundancy computations **/
  private def normalizeRankingDF(partialWeights: RDD[(Int, Float)]) = {
      val maxRelief = partialWeights.values.max()
      val minRelief = partialWeights.values.min()
      partialWeights.mapValues{ x => (x - minRelief) / (maxRelief - minRelief)}
  }  
  
  private def approxNNByPartition(
      modelDataset: Dataset[_],
      bModelQuery: Broadcast[Array[Row]],
      k: Int): RDD[(Long, Map[Int, Iterable[Int]])] = {
    
    case class Localization(part: Int, index: Int)
    val sc = modelDataset.sparkSession.sparkContext
    val icol = $(inputCol) // Use local variables to avoid shuffle the entire object to each map.
    val input: RDD[Row] = modelDataset.select(icol).rdd
    
    val neighbors: RDD[(Long, Map[Int, Iterable[Int]])] = input.mapPartitionsWithIndex { 
      case (pindex, it) => 
          // Initialize the map composed by the priority queue and the central element's ID
          val query = bModelQuery.value // Query set. Fields: "UniqueID", $(inputCol), $(labelCol)
          val ordering = Ordering[Float].on[(Float, Localization)](_._1).reverse// BPQ needs reverse ordering 
          // BPQ to sort from closer neighbors to farther (to be fullfilled below)
          val neighbors = query.map { r => r.getAs[Long]("UniqueID") -> 
              new BPQ[(Float, Localization)](k)(ordering) 
            }   
      
          var i = 0 // index for local elements in this partition.
          while(it.hasNext) { // local elements
            val inputNeig = it.next.getAs[Vector](icol)
              (0 until query.size).foreach { j => // query elements broadcasted
                 val distance = Math.sqrt(Vectors.sqdist(query(j).getAs[Vector](icol), inputNeig)).toFloat
                 neighbors(j)._2 += distance -> Localization(pindex.toShort, i)    
               }
            i += 1              
          }            
          neighbors.toIterator
      }.reduceByKey(_ ++= _).mapValues(
          _.map(l => Localization.unapply(l._2).get).groupBy(_._1).mapValues(_.map(_._2)).map(identity)) 
      // Select nearest neighbors from all partitions. For each query element, create a map (partitionID, list of local indices)   
      // map(identity) needed to fix bug: https://issues.scala-lang.org/browse/SI-7005
    neighbors
  }
  
  /* Enum class */
  object MOD {
    val sameClass = 0
    val otherClass = 1
  }
  
  private def computeReliefWeights (
      modelDataset: Dataset[_],
      bModelQuery: Broadcast[Array[Row]],
      bNeighborsTable: Broadcast[Map[Long, Map[Int, Iterable[Int]]]],
      topFeatures: Set[Int],
      priorClass: Map[Double, Float],
      nFeat: Int,
      nElems: Long) = {
    
    // Auxiliary vars
    val sc = modelDataset.sparkSession.sparkContext
    val label2Num: Map[Double, Int] = priorClass.zipWithIndex.map{case (k, v) => k._1 -> v} // Output labels to numeric indices
    val idxPriorClass: Map[Int, Float] = priorClass.zipWithIndex.map{case (k, v) => v -> k._2} // Proportion by class
    
    // Accumulators and broadcasted vars
    val accRedMarginal = new VectorAccumulator(nFeat, sparse = false); sc.register(accRedMarginal, "marginal")
    val accRedJoint = new MatrixAccumulator(nFeat, nFeat, sparse = false); sc.register(accRedJoint, "joint")
    val totalInteractions = sc.longAccumulator("totalInteractions")
    val bClassCounter = new VectorAccumulator(label2Num.size * 2, sparse = false); sc.register(bClassCounter, "classCounter")
    val bTF: Broadcast[Set[Int]] = sc.broadcast(topFeatures)
    
    // Use local vars to avoid sending the entire object to mappers
    val isCont = !$(discreteData); val lowerDistanceTh = $(lowerDistanceThreshold); val icol = $(inputCol); val lcol = $(labelCol)
        
    // Left side: relevance, right side: redundancy joint
    val rawReliefWeights: RDD[(Int, (BDV[Float], Vector))] = 
      modelDataset.rdd.mapPartitionsWithIndex { case(pindex, it) =>
        // last position is reserved to negative weights from central instances.
        val localExamples = it.toArray
        val r = new scala.util.Random($(seed))
        val marginal = BDV.zeros[Double](nFeat) // Marginal proportions computed for [redundancy]
        // Matrix with # features columns and 2 * # labels rows for [relevance]
        // Labels are duplicated to distinguish between instances that do/don't share class with query
        // Left side: relevance, right side: redundancy joint
        val reliefWeights: BDV[(BDV[Float], BDV[Double])] = 
          BDV.fill[(BDV[Float], BDV[Double])](nFeat)(
              (BDV.zeros[Float](label2Num.size * 2), BDV.zeros[Double](nFeat)))
        val classCounter: BDV[Double] = BDV.zeros[Double](label2Num.size * 2)      
        
        // Data are assumed to be scaled to have 0 mean, and 1 std. Checks are costly, so they're skipped
        val vote = if(isCont) (d: Double) => 1 - math.min(6.0, d) / 6.0 else (d: Double) => Double.MinPositiveValue
        val pcounter = Array.fill(nFeat)(0.0d) // Store individual votes for each feature
        val jointVote = if(isCont) (i1: Int, i2: Int) => (pcounter(i1) + pcounter(i2)) / 2 else // Joint votes
                      (i1: Int, _: Int) => pcounter(i1)
                      
        // Compute redundancy and relevancy contributions among the query set and the neighborhood set
        bModelQuery.value.map{ row =>
            val qid = row.getAs[Long]("UniqueID"); val qinput = row.getAs[Vector](icol)
            val qlabel = row.getAs[Double](lcol)
            bNeighborsTable.value.get(qid) match { 
              case Some(localMap) =>
                localMap.get(pindex) match {
                  case Some(neighbors) =>
                    // Distance values greater than the threshold will be considered as outlier (lowerdistanceTh * 6 times std deviation)
                    val outlierThreshold = if(isCont) 6 * (1 - (lowerDistanceTh + r.nextFloat() * lowerDistanceTh)) else 0.0f
                    neighbors.map{ lidx =>
                      val Row(ninput: Vector, nlabel: Double) = localExamples(lidx)
                      // If both share class contribution is placed in row = 1, if not in row = 2
                      val modIdx = if(nlabel == qlabel) label2Num.size * MOD.sameClass else label2Num.size * MOD.otherClass
                      classCounter(label2Num(nlabel) + modIdx) += 1
                            
                      qinput.foreachActive{ (index, value) =>
                         val fdistance = math.abs(value - ninput(index))
                         //// Annotate RELIEF computations
                         reliefWeights(index)._1(label2Num(nlabel) + modIdx) += fdistance.toFloat
                         ////// mCR functionality (collision detection)
                         // The closer the distance, the more probable is the co-occurrence.
                         if(fdistance <= outlierThreshold){
                            val contribution = vote(fdistance)
                            marginal(index) += contribution
                            pcounter(index) = contribution
                            // If the feature was relevant in the prev step, we compute its redundancy relationships.
                            if(bTF.value.contains(index)){ 
                              // Annotate co-occurrences between the new feature and the remaining features.
                              qinput.foreachActive{ (i2, _) =>
                                if(i2 != index){
                                  reliefWeights(index)._2(i2) += jointVote(index, i2)
                                  reliefWeights(i2)._2(index) += jointVote(index, i2)  
                                }                                
                              }
                            }
                         }
                        }
                    }
                  case None => /* Do nothing */
                }                
              case None =>
                System.err.println("Instance does not found in the table")
            }
      }
      // update accumulated matrices 
      accRedMarginal.add(marginal)
      totalInteractions.add(classCounter.sum.toLong)
      bClassCounter.add(classCounter)
      // Left side: relevance, right side: redundancy joint 
      reliefWeights.iterator      
    }.reduceByKey{ case((r1, j1), (r2, j2)) => (r1 + r2, j1 + j2) }.mapValues{ 
          case (rel, red) => rel -> Vectors.fromBreeze(red)
        }
      
    val weights: RDD[(Int, (Float, Vector))] = 
        aggregateWeightsByFeat(rawReliefWeights, bClassCounter, idxPriorClass)
         
    (weights, accRedMarginal, totalInteractions)
  }
  
  private def computeReliefWeightsSparse (
      modelDataset: Dataset[_],
      bModelQuery: Broadcast[Array[Row]],
      bNeighborsTable: Broadcast[Map[Long, Map[Int, Iterable[Int]]]],
      topFeatures: Set[Int],
      priorClass: Map[Double, Float],
      nFeat: Int,
      nElems: Long
      ) = {
    
      // Auxiliary vars
      val sc = modelDataset.sparkSession.sparkContext
      val label2Num: Map[Double, Int] = priorClass.zipWithIndex.map{case (k, v) => k._1 -> v} // Output labels to numeric indices
      val idxPriorClass: Map[Int, Float] = priorClass.zipWithIndex.map{case (k, v) => v -> k._2} // Proportion by class
      
      // Accumulators and broadcasted vars
      val accRedMarginal = new VectorAccumulator(nFeat, sparse = true); sc.register(accRedMarginal, "marginal")
      val totalInteractions = sc.longAccumulator("totalInteractions")
      val bClassCounter = new VectorAccumulator(label2Num.size * 2, sparse = false); sc.register(bClassCounter, "classCounter")
      val bTF: Broadcast[Set[Int]] = sc.broadcast(topFeatures)
    
      // Use local vars to avoid sending the entire object to mappers
      val isCont = !$(discreteData); val lowerDistanceTh = $(lowerDistanceThreshold); val icol = $(inputCol); val lcol = $(labelCol)
      val lseed = $(seed)
      
      // Left side: relevance, right side: redundancy joint
      val rawReliefWeights: RDD[(Int, (BDV[Float], Vector))] = 
        modelDataset.rdd.mapPartitionsWithIndex { case(pindex, it) =>
          // last position is reserved to negative weights from central instances.
          val localExamples = it.toArray
          val marginal = new VectorBuilder[Double](nFeat) // Marginal proportions computed for [redundancy]
          // Left side: relevance, right side: redundancy joint
          val reliefWeights = new HashMap[Int, (BDV[Float], VectorBuilder[Double])]
          // Vector with 2 * # labels positions for [relevance]
          // Labels are duplicated to distinguish between instances that do/don't share class with query
          val classCounter = BDV.zeros[Double](label2Num.size * 2)        
          val r = new scala.util.Random(lseed)
          // Data are assumed to be scaled to have 0 mean, and 1 std
          val vote = Double.MinPositiveValue
          
          bModelQuery.value.map{ row =>
              val qid = row.getAs[Long]("UniqueID"); val qinput = row.getAs[SparseVector](icol) 
              val qlabel = row.getAs[Double](lcol)
              bNeighborsTable.value.get(qid) match { 
                case Some(localMap) =>
                  localMap.get(pindex) match {
                    case Some(neighbors) =>
                      val rnumber = r.nextFloat()
                      // Distance values greater than the threshold will be considered as outliers (lowerdistanceTh * 6 times std deviation)
                      val distanceThreshold = if(isCont) 6 * (1 - (lowerDistanceTh + rnumber * lowerDistanceTh)) else 0.0f
                      neighbors.map{ lidx =>
                        val Row(ninput: SparseVector, nlabel: Double) = localExamples(lidx)
                        val labelIndex = label2Num.get(nlabel.toFloat).get
                        val mod = if(nlabel == qlabel) label2Num.size * MOD.sameClass else label2Num.size * MOD.otherClass
                        classCounter(labelIndex + mod) += 1
                        // Perform standard loop to fetch sparse vectors
                        val v1Indices = qinput.indices; val nnzv1 = v1Indices.length; var kv1 = 0
                        val v2Indices = ninput.indices; val nnzv2 = v2Indices.length; var kv2 = 0
                        
                        while (kv1 < nnzv1 || kv2 < nnzv2) {
                          var score = 0.0; var index = kv1                
                          if (kv2 >= nnzv2 || (kv1 < nnzv1 && v1Indices(kv1) < v2Indices(kv2))) {
                            score = qinput.values(kv1); kv1 += 1; index = kv1
                          } else if (kv1 >= nnzv1 || (kv2 < nnzv2 && v2Indices(kv2) < v1Indices(kv1))) {
                            score = ninput.values(kv2); kv2 += 1; index = kv2
                          } else {
                            score = qinput.values(kv1) - ninput.values(kv2); kv1 += 1; kv2 += 1; index = kv1
                          }
                          
                          val fdistance = math.abs(score)
                          //// RELIEF Computations
                          // Update hashMap with contributions for each featurek
                          val stats = reliefWeights.get(index) match {
                            case Some(st) => 
                              st._1(labelIndex + mod) += fdistance.toFloat
                              st
                            case None => 
                              val st = (BDV.zeros[Float](label2Num.size * 2), new VectorBuilder[Double](nFeat))
                              reliefWeights += index -> st
                              st._1(labelIndex + mod) += fdistance.toFloat                              
                              st
                          }
                          
                          //// Check co-occurrences for redundancy
                          // The closer the distance, the more probable.
                           if(fdistance <= distanceThreshold){
                              marginal.add(index, vote)
                              //pcounter(index) = contribution
                              if(bTF.value.contains(index)){ // Relevant, the feature is added to the main group and update auxiliar feat's.
                                qinput.foreachActive{ (i2, _) =>
                                  if(i2 != index){
                                    stats._2.add(i2, vote)
                                    reliefWeights.get(i2).get._2.add(index, vote)
                                  }                                
                                }                            
                              }
                          }                       
                        }
                  }
                case None => /* Do nothing */
              }
              case None =>
                  System.err.println("Instance does not found in the table")
            }
        }
        // update accumulated matrices  
        accRedMarginal.add(marginal.toSparseVector)        
        totalInteractions.add(classCounter.sum.toLong)
        bClassCounter.add(classCounter)
        reliefWeights.mapValues{ case(relief, joint) => relief -> joint.toSparseVector }.iterator
        
      }.reduceByKey{ case((r1, j1), (r2, j2)) => (r1 + r2, j1 + j2) }.mapValues{ 
          case (rel, red) => rel -> Vectors.fromBreeze(red)
        }
      
      val weights: RDD[(Int, (Float, Vector))] = 
        aggregateWeightsByFeat(rawReliefWeights, bClassCounter, idxPriorClass)
      
      (weights, accRedMarginal, totalInteractions)
  }
  
  /** Once performed partition-wise computations, aggregate contributions to create feature-based scores **/
  private def aggregateWeightsByFeat(
      rawReliefWeights: RDD[(Int, (BDV[Float], Vector))], 
      bClassCounter: VectorAccumulator,
      idxPriorClass: Map[Int, Float]): RDD[(Int, (Float, Vector))] = {
    
    val cc: BV[Double] = bClassCounter.value
    // Aggregate final relevance weights and joint contributions to create redundancy scores by feature
    val reliefWeights: RDD[(Int, (Float, Vector))] = rawReliefWeights.mapValues { 
      case (relevanceByClass, redudancyJoint) =>
       val nClasses = idxPriorClass.size; var sum = 0.0f
       relevanceByClass.foreachPair{ case (classMod, value) =>
           val contrib = if(cc(classMod) > 0) {
               val sign = if(classMod / nClasses == MOD.sameClass) -1 else 1 // class is shared?
               // negative/positive contribution + class weight + value / number of neighbors implied
               sign * idxPriorClass.get(classMod % nClasses).get * (value / cc(classMod).toFloat)
             } else {
               0.0f
             }
           sum += contrib
       }
       sum -> redudancyJoint
    }
    
    reliefWeights
  }
  
  /** Transform co-occurrences to redundancy scores using marginal and joint annotations **/
  private def computeRedudancy(
      weights: RDD[(Int, (Float, Vector))], 
      rawMarginal: BV[Double], 
      total: Long, 
      nFeat: Int, 
      batchPerc: Double, 
      sparse: Boolean): RDD[(Int, (Float, Vector))] = {
    // Now compute redundancy based on collisions and normalize it
    val factor = if($(discreteData) || sparse) Double.MinPositiveValue else 1.0
    val marginal: BV[Double] = rawMarginal.mapActiveValues(_ / (total * factor))
    val jointTotal = total * factor * (1 - $(estimationRatio) * batchPerc) // the total number of co-ocurrences omitting those in the first batch
    
    // Function that compute joint entropy between two features given a joint value
    val applyEntropy = (i1: Int, i2: Int, value: Double) => {
      val jprob = value / jointTotal
      val red = jprob * log2(jprob / (marginal(i1) * marginal(i2)))  
      if(!red.isNaN()) red else 0.0d
    }
    
    // Apply the previous function to each pair of features in jointRedundancy
    val entropyWeights: RDD[(Int, (Float, Vector))] = weights.map{ 
      case (i1, (relevance, jointRedundancy)) => 
        val res = jointRedundancy.asBreeze match {   
          case sv: BSV[Double] => 
            sv.mapActivePairs{case (i2, value) => applyEntropy(i1, i2, value)}
          case dv: BDV[Double] => 
            dv.mapActivePairs{case (i2, value) => applyEntropy(i1, i2, value)}
        }
        i1 -> (relevance, Vectors.fromBreeze(res))
    }
    
    val maxRed = entropyWeights.values.map{ case (_, joint) => joint.asBreeze.max}.max
    val minRed = entropyWeights.values.map{ case (_, joint) => joint.asBreeze.min}.min
    
    // Finally, normalize redundancy scores with minmax procedure
    val normalizedWeights: RDD[(Int, (Float, Vector))] = entropyWeights.mapValues{ 
      case (w, joint) => 
        val res = joint.asBreeze match {
          case sv: BSV[Double] => 
            sv.mapActiveValues(e => (e - minRed) / (maxRed - minRed))
          case dv: BDV[Double] => 
            dv.mapActiveValues(e => (e - minRed) / (maxRed - minRed))
        }
        w -> Vectors.fromBreeze(res)
    }
    
    normalizedWeights
  }

  // Case class for criteria/feature
  case class F(feat: Int, crit: FeatureScore)
  
  /** Greedy process to select best N features according to relevance and redundancy scores **/
  private def selectFeatures(reliefRanking: RDD[(Int, (Float, Vector))], nFeat: Int): (Seq[F], Seq[F]) = {
    
    // Initialize standard RELIEF only with the relevance scores
    implicit def scoreOrder: Ordering[(Int, (Float, Vector))] = Ordering.by{ r => (-r._2._1, r._1) }
    val stdRanking: Seq[F] = reliefRanking.takeOrdered($(numTopFeatures))(scoreOrder).map{ 
          case (feat, (crit, _)) => F(feat, new FeatureScore().init(crit.toFloat)) }.toSeq
    val reliefRed: Array[(Int, (FeatureScore, Vector))] = reliefRanking.mapValues{ 
            case(score, redundancy) => new FeatureScore().init(score.toFloat) -> redundancy
          }.collect()
    val pool: Array[Option[(FeatureScore, Vector)]] = Array.fill(nFeat)(None)
    reliefRed.foreach{ case (id, score) => pool(id) = Some(score) }
    // Get the best ranked feature and initialize the set of selected features with it
    var selected: Seq[F] = Seq(stdRanking.head)
    pool(selected.head.feat).get._1.valid = false // Can't be selected again
    var moreFeat = true
    
    // Greedy iterative process to select numTopFeatures according to the trade-off relevance vs. redundancy
    while (selected.size < $(numTopFeatures) && moreFeat) {

      // Update non-selected features by computing redundancy between them and the last selected feat.
      pool(selected.head.feat).get._2.foreachActive{ 
         case(k, v) => 
           if(pool(k).get._1.valid) 
             pool(k).get._1.update(v.toFloat)
      }
      
      // select the next best feature and remove from the non-selected set of features
      var max = new FeatureScore().init(Float.NegativeInfinity)
      val maxscore = max.score; var maxi = -1
      (0 until pool.size).foreach{ i => 
        pool(i) match {
          case Some((crit, _)) => 
            if(crit.valid && crit.score > max.score){
              maxi = i; max = crit
            } 
          case None => /* Do nothing */
        }
      }
      
      if(maxi > -1){
        selected = F(maxi, max) +: selected
        max.valid = false
      } else {
        moreFeat = false
      }
    }
    (selected.reverse, stdRanking)  
  }
  
  /** Inner class that annotates changes in relevance and redundancy in each feature during the selection process **/
  class FeatureScore extends Serializable {
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
    override def toString() = {
      "%.8f".format(score) + "\t" +
      "%.8f".format(relevance) + "\t" +
      "%.8f".format(redundance)
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
    private val stdSelection: Array[Int],
    private val redundancySelection: Array[Int]
  ) extends Model[ReliefFRSelectorModel] with ReliefFRSelectorParams with MLWritable {

  import ReliefFRSelectorModel._
  
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setInputCol(value: String): this.type = set(inputCol, value)
  def setRedundancyRemoval(value: Boolean): this.type = set(redundancyRemoval, value)
  
  private var selectionSize: Int = getSelectedFeatures().length
  def setReducedSubset(s: Int): this.type =  {
    if(s > 0 && s <= getSelectedFeatures().length){
      selectionSize = s
    } else {
      System.err.println("The number of features in the subset must" +
        " be lower than the total number and greater than 0")
    }
    this
  }
  
  def getReducedSubsetParam(): Int = selectionSize
  def getSelectedFeatures(): Array[Int] = if($(redundancyRemoval)) redundancySelection else stdSelection

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformedSchema = transformSchema(dataset.schema, logging = true)
    val newField = transformedSchema.last

    // TODO: Make the transformer natively in ml framework to avoid extra conversion.
    val selection: Array[Int] = getSelectedFeatures().slice(0, selectionSize).sorted
    // sfeat must be ordered asc
    val transformer: Vector => Vector = v =>  FeatureSelectionUtils.compress(OldVectors.fromML(v), selection).asML
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
      origAttrGroup.attributes.get.zipWithIndex.filter(x => stdSelection.contains(x._2)).map(_._1)
    } else {
      Array.fill[Attribute](stdSelection.size)(NumericAttribute.defaultAttr)
    }
    val newAttributeGroup = new AttributeGroup($(outputCol), featureAttributes)
    newAttributeGroup.toStructField()
  }

  override def copy(extra: ParamMap): ReliefFRSelectorModel = {
    val copied = new ReliefFRSelectorModel(uid, stdSelection, redundancySelection)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new ReliefFRSelectorModelWriter(this)
}

@Since("1.6.0")
object ReliefFRSelectorModel extends MLReadable[ReliefFRSelectorModel] {

  private[ReliefFRSelectorModel] class ReliefFRSelectorModelWriter(instance: ReliefFRSelectorModel) extends MLWriter {

    private case class Data(stdSelection: Seq[Int], redundancySelection: Seq[Int])

    // Save model to parquet file format (2 different sequences)
    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.stdSelection.toSeq, instance.redundancySelection.toSeq)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class ReliefFRSelectorModelReader extends MLReader[ReliefFRSelectorModel] {

    private val className = classOf[ReliefFRSelectorModel].getName

    override def load(path: String): ReliefFRSelectorModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("stdSelection", "redundancySelection").head()
      val reliefFeatures = data.getAs[Seq[Int]](0).toArray
      val reliefColFeatures = data.getAs[Seq[Int]](1).toArray
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
