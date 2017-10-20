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

/**
 * Params for [[ReliefClsSelector]] and [[ReliefClsSelectorModel]].
 */
private[feature] trait ReliefClsSelectorParams extends Params
    with HasInputCol with HasOutputCol with HasLabelCol with HasSeed {

  /**
   * Information Theoretic criterion used to rank the features. The default value is the criterion mRMR.
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
  setDefault(numNeighbors -> 25)
  
  final val estimationRatio: DoubleParam = new DoubleParam(this, "estimationRatio", "", ParamValidators.inRange(0,1))
  setDefault(estimationRatio -> 0.25)
  
  final val batchSize: DoubleParam = new DoubleParam(this, "batchSize", "", ParamValidators.inRange(0,1))
  setDefault(batchSize -> 0.1)
  
  final val queryStep: IntParam = new IntParam(this, "queryStep", "", ParamValidators.gtEq(1))
  setDefault(queryStep -> 2)

}

/**
 * :: Experimental ::
 * Chi-Squared feature selection, which selects categorical features to use for predicting a
 * categorical label.
 */
@Experimental
final class ReliefClsSelector @Since("1.6.0") (@Since("1.6.0") override val uid: String = Identifiable.randomUID("ReliefClsSelector"))
    extends Estimator[ReliefClsSelectorModel] with ReliefClsSelectorParams with DefaultParamsWritable {

  /** @group setParam */
  @Since("1.6.0")
  def setNumTopFeatures(value: Int): this.type = set(numTopFeatures, value)

  /** @group setParam */
  @Since("1.6.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  @Since("1.6.0")
  def setLabelCol(value: String): this.type = set(labelCol, value)

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): ReliefClsSelectorModel = {
    transformSchema(dataset.schema, logging = true)
    val input: RDD[OldLabeledPoint] =
      dataset.select(col($(labelCol)).cast(DoubleType), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          OldLabeledPoint(label, OldVectors.fromML(features))
      }.cache()

    val ReliefClsSelector = new feature.ReliefClsSelector(
      new ReliefClsCriterionFactory($(selectCriterion)),
      $(numTopFeatures),
      $(nPartitions)
    ).fit(input)
    copyValues(new ReliefClsSelectorModel(uid, ReliefClsSelector).setParent(this))
  }
  
  
  // TODO: Fix the MultiProbe NN Search in SPARK-18454
  private[feature] def selectFeatures(
      dataset: Dataset[_]) = {
    
    val continuous = true
    
    // Get Hash Value of the key
    
    val brp = new BucketedRandomProjectionLSH()
      .setNumHashTables($(numHashTables))
      .setInputCol($(inputCol))
      .setOutputCol("hashCol")
      .setBucketLength($(bucketLength))
      .setSignatureSize($(signatureSize))
      .setSeed($(seed))
    val model = brp.fit(dataset)
    
    val modelDataset: DataFrame = if (!dataset.columns.contains($(outputCol))) {
        model.transform(dataset)
      } else {
        dataset.toDF()
      }
    val weights = Array.fill(($(batchSize) * 100).toInt)($(estimationRatio) * $(batchSize))
    val batches = modelDataset.randomSplit(weights, $(seed))
    
    // Get some basic information about the dataset
    val sc = modelDataset.sparkSession.sparkContext
    val nFeat = modelDataset.select($(inputCol)).head().getAs[Vector](0).size
    val nElems = modelDataset.count()
    val priorClass = modelDataset.select($(labelCol)).rdd.map{ case Row(label: Double) => label.toFloat }
        .countByValue()
        .mapValues(_ / nElems)
        .map(identity).toMap
  
    var featureWeights: Array[(Int, Float)] = Array.empty
    var redundancyMatrix: DenseMatrix[Float] = DenseMatrix.zeros(nFeat, nFeat)
    for(batchIndex <- 0 until batches.size) {
      val query = batches(batchIndex)
      val modelQuery: DataFrame = if (!query.columns.contains($(outputCol))) {
        model.transform(query)
      } else {
        dataset.toDF()
      }  
    
      // Index query objects and compute the table that indicates where are located its neighbors
      val idxModelQuery = modelQuery.withColumn("UniqueID", monotonically_increasing_id).cache()
      val bFullQuery: Broadcast[Array[Row]] = sc.broadcast(
          idxModelQuery.select("UniqueID", $(inputCol), $(labelCol), $(outputCol)).collect())
      val neighbors: RDD[(Long, Map[Short, Iterable[Int]])] = approxNNByPartition(modelDataset, 
          bFullQuery, $(numNeighbors) * priorClass.size, $(queryStep))
      val bNeighborsTable: Broadcast[Map[Long, Map[Short, Iterable[Int]]]] = 
        sc.broadcast(neighbors.collectAsMap().toMap)
  
      val partialResults = computeReliefWeights(
          modelDataset.select($(inputCol), $(labelCol)), bFullQuery, bNeighborsTable, 
          priorClass, nFeat, nElems, continuous, featureWeights.toMap)
      featureWeights = partialResults._1
      redundancyMatrix = partialResults._2
    }
    
        
  }
  
  private def computeReliefWeights (
      modelDataset: Dataset[_],
      bModelQuery: Broadcast[Array[Row]],
      bNeighborsTable: Broadcast[Map[Long, Map[Short, Iterable[Int]]]],
      priorClass: Map[Float, Long],
      nFeat: Int,
      nElems: Long,
      isCont: Boolean,
      oldReliefWeights: Map[Int, Float] = Map.empty
      ) = {
    
     // Initialize accumulators for RELIEF+Collision computation
    val sc = modelDataset.sparkSession.sparkContext
    val accMarginal = new DoubleVectorAccumulator(nFeat); sc.register(accMarginal, "marginal")
    val accJoint = new MatrixAccumulator(nFeat, nFeat);  sc.register(accJoint, "joint")
    val neighborClassCount = new LongVectorAccumulator(nFeat + 1); sc.register(neighborClassCount, "classCount")
    val lowerTh = 0.75f
    val label2Num = priorClass.zipWithIndex.map{case (k, v) => k._1 -> v.toShort}
    val bLabel2Num = sc.broadcast(label2Num) 
    val bPriorClass = sc.broadcast(priorClass)
    val bOldRW = sc.broadcast(oldReliefWeights)
    
    val rawReliefWeights = modelDataset.rdd.mapPartitionsWithIndex { 
      case(pindex, it) =>
        val localExamples = it.toArray
        val query = bModelQuery.value
        val table = bNeighborsTable.value
        val labelConversion = bLabel2Num.value
        // last position is reserved to negative weights from central instances.
        val reliefWeights = Array.fill(nFeat, nFeat + 1)(0.0f) 
        val marginal = breeze.linalg.DenseVector.zeros[Double](nFeat)
        val joint = breeze.linalg.DenseMatrix.zeros[Double](nFeat, nFeat)
        val classCounter = breeze.linalg.DenseVector.zeros[Long](nFeat + 1)
        val r = new scala.util.Random($(seed))
        // Data are assumed to be scaled to have 0 mean, and 1 std
        val vote = if(isCont) (d: Double) => 1 - math.min(6.0, d) / 6.0 else (d: Double) => Double.MinPositiveValue
        val ow = if(bOldRW.value.isEmpty) (0 until nFeat).map(i => i -> 1.0f).toMap else bOldRW.value
        
        query.map{ case Row(qid: Long, qinput: Vector, qlabel: Double, _) =>
            
            val condition = if(isCont) 6 * (1 - (lowerTh + r.nextFloat() * lowerTh)) else 0
            val condition2 = ow.mapValues{ _ >= r.nextFloat()}
        
            table.get(qid) match { case Some(localMap) =>
                val neighbors = localMap.getOrElse(pindex.toShort, Iterable.empty)
                neighbors.map{ lidx =>
                  val Row(ninput: Vector, nlabel: Double) = localExamples(lidx)
                  var collisioned = Queue[Int]() // annotate the features matched up to now
                  val pcounter = Array.fill(nFeat)(0.0d) // count the strength of collision in each feature
                  val jvote = if(isCont) (i1: Int, i2: Int) => (pcounter(i1) + pcounter(i2)) / 2 else 
                    (i1: Int, i2: Int) => pcounter(i1)
                  val labelIndex = if(nlabel != qlabel) labelConversion.get(nlabel.toFloat).get else labelConversion.size
                  classCounter(labelIndex) += 1  
                        
                  qinput.foreachActive{ (index, value) =>
                     val fdistance = math.abs(value - ninput(index))
                     //// RELIEF Computations
                     reliefWeights(index)(labelIndex) += fdistance.toFloat
                     //// Collision-based computations
                     // The closer the distance, the more probable.
                     // The higher the score in the previous rank, the more probable.
                     if(fdistance <= condition && condition2.getOrElse(index, true)){
                        val contribution = vote(fdistance)
                        marginal(index) += contribution
                        pcounter(index) = contribution
                        val fit = collisioned.iterator
                        while(fit.hasNext){
                          val i2 = fit.next
                          joint(i2, index) += jvote(index, i2)
                        }                         
                        collisioned += index
                     }
                  }
                }
              case None =>
                System.err.println("Instance does not found in the table")
            }
            // update accumulated matrices  
            accMarginal.add(marginal)
            accJoint.add(joint) 
            neighborClassCount.add(classCounter)
      }
      reliefWeights.zipWithIndex.map(_.swap).toIterator
    }.reduceByKey(_ ++ _)
    
    val clsC = neighborClassCount.value
    
    val reliefWeights = rawReliefWeights.mapValues { weights => 
      val labelConversion = bLabel2Num.value
      val priorProp = bPriorClass.value
      var sum = labelConversion.map{ case(label, index) =>
        priorProp.get(label).get * weights(index) / clsC(index)
      }.sum  
      sum -= weights(weights.size) / clsC(weights.size)
      sum / nElems
    }.cache
    
    val maxRelief = reliefWeights.values.max()
    val minRelief = reliefWeights.values.min()
    val normalizedRelief = reliefWeights.mapValues(score => ((score - minRelief) / (maxRelief - minRelief)).toFloat).collect
    
    val factor = if(isCont) 1.0 else Double.MinPositiveValue
    val total = clsC.sum
    val marginal = accMarginal.value.toDenseVector.mapPairs{ case(_, e) => e /  (total * factor) }  
    val joint = accJoint.value.toDenseMatrix.mapPairs{ case(_, e) => e /  (total * factor) }  
    
    // Compute mutual information using collisions with and without class
    val redundancyMatrix = breeze.linalg.DenseMatrix.zeros[Float](nFeat, nFeat)
    joint.activeIterator.foreach { case((i1,i2), value) =>
      if(i1 < i2) {
        val red = (value * log2(value / (marginal(i1) * marginal(i2)))).toFloat              
        redundancyMatrix(i1, i2) = red; redundancyMatrix(i2, i1) = red        
      }        
    }
    
    val maxRed = breeze.linalg.max(redundancyMatrix)
    val minRed = breeze.linalg.min(redundancyMatrix)
    val normRedundancyMatrix = redundancyMatrix.mapValues{ e => ((e - minRed) / (maxRed - minRed)).toFloat }
    
    (normalizedRelief, normRedundancyMatrix)
  }
  
  
  
    // TODO: Fix the MultiProbe NN Search in SPARK-18454
  private def approxNNByPartition(
      modelDataset: Dataset[_],
      bModelQuery: Broadcast[Array[Row]],
      k: Int,
      step: Int = 3) = {
    
    case class Localization(part: Short, index: Int)
    val sc = modelDataset.sparkSession.sparkContext
    val hashThDistance = (h1: Array[Vector], h2: Array[Vector]) => model.hashThresholdedDistance(h1, h2, step)
    
    val neighbors = modelDataset.select($(outputCol)).rdd.mapPartitionsWithIndex { 
        case (pindex, it) => 
        
          val ordering = Ordering[Float].on[(Float, Localization)](-_._1)   
          val query = bModelQuery.value
          val neighbors = query.map { case Row(id: Long, _) => 
            id -> new BoundedPriorityQueue[(Float, Localization)](k)(ordering)
          }   
      
          var i = 0
          while(it.hasNext) {
            val Row(hashNeig: Array[Vector]) = it.next
            (0 until query.size).foreach { j => 
               val Row(_, hashQuery: Array[Vector]) = query(j) 
               val dist = hashThDistance(hashQuery, hashNeig)
               if(dist < Float.PositiveInfinity)
                 neighbors(j)._2 += dist.toFloat -> Localization(pindex.toShort, i)
            }
            i += 1              
          }            
          neighbors.toIterator
      }.reduceByKey(_ ++= _).mapValues(
          _.map(l => Localization.unapply(l._2).get).groupBy(_._1).mapValues(_.map(_._2)).toMap)
    neighbors
  }
  
  private def log2(x: Double) = { math.log(x) / math.log(2) }
  

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(labelCol), DoubleType)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }

  override def copy(extra: ParamMap): ReliefClsSelector = defaultCopy(extra)
}

@Since("1.6.0")
object ReliefClsSelector extends DefaultParamsReadable[ReliefClsSelector] {

  @Since("1.6.0")
  override def load(path: String): ReliefClsSelector = super.load(path)
}

/**
 * :: Experimental ::
 * Model fitted by [[ReliefClsSelector]].
 */
@Experimental
final class ReliefClsSelectorModel private[ml] (
  @Since("1.6.0") override val uid: String,
  private val ReliefClsSelector: feature.ReliefClsSelectorModel
)
    extends Model[ReliefClsSelectorModel] with ReliefClsSelectorParams with MLWritable {

  import ReliefClsSelectorModel._

  /** list of indices to select (filter). Must be ordered asc */
  val selectedFeatures: Array[Int] = ReliefClsSelector.selectedFeatures
  
  val redMap: HashMap[Int, Array[(Int, (Float, Float))]] = ReliefClsSelector.redMap

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformedSchema = transformSchema(dataset.schema, logging = true)
    val newField = transformedSchema.last

    // TODO: Make the transformer natively in ml framework to avoid extra conversion.
    val transformer: Vector => Vector = v => ReliefClsSelector.transform(OldVectors.fromML(v)).asML

    val selector = udf(transformer)

    dataset.withColumn($(outputCol), selector(col($(featuresCol))), newField.metadata)
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    val newField = prepOutputField(schema)
    val outputFields = schema.fields :+ newField
    StructType(outputFields)
  }

  /**
   * Prepare the output column field, including per-feature metadata.
   */
  private def prepOutputField(schema: StructType): StructField = {
    val selector = ReliefClsSelector.selectedFeatures.toSet
    val origAttrGroup = AttributeGroup.fromStructField(schema($(featuresCol)))
    val featureAttributes: Array[Attribute] = if (origAttrGroup.attributes.nonEmpty) {
      origAttrGroup.attributes.get.zipWithIndex.filter(x => selector.contains(x._2)).map(_._1)
    } else {
      Array.fill[Attribute](selector.size)(NumericAttribute.defaultAttr)
    }
    val newAttributeGroup = new AttributeGroup($(outputCol), featureAttributes)
    newAttributeGroup.toStructField()
  }

  override def copy(extra: ParamMap): ReliefClsSelectorModel = {
    val copied = new ReliefClsSelectorModel(uid, ReliefClsSelector)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MLWriter = new ReliefClsSelectorModelWriter(this)
}

@Since("1.6.0")
object ReliefClsSelectorModel extends MLReadable[ReliefClsSelectorModel] {

  private[ReliefClsSelectorModel] class ReliefClsSelectorModelWriter(instance: ReliefClsSelectorModel) extends MLWriter {

    private case class Data(selectedFeatures: Seq[Int])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.selectedFeatures.toSeq)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class ReliefClsSelectorModelReader extends MLReader[ReliefClsSelectorModel] {

    private val className = classOf[ReliefClsSelectorModel].getName

    override def load(path: String): ReliefClsSelectorModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath).select("selectedFeatures").head()
      val selectedFeatures = data.getAs[Seq[Int]](0).toArray
      val oldModel = new feature.ReliefClsSelectorModel(selectedFeatures)
      val model = new ReliefClsSelectorModel(metadata.uid, oldModel)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  @Since("1.6.0")
  override def read: MLReader[ReliefClsSelectorModel] = new ReliefClsSelectorModelReader

  @Since("1.6.0")
  override def load(path: String): ReliefClsSelectorModel = super.load(path)
}
