package org.apache.spark.ml.feature

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.sql.Row
import scala.collection.mutable.TreeSet
import org.apache.spark.util.LongAccumulator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.SparkConf
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.Pipeline
import breeze.linalg.functions.euclideanDistance
import breeze.stats.MeanAndVariance
import breeze.stats.DescriptiveStats
import breeze.linalg.mapValues
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.feature.InfoThCriterion
import org.apache.spark.mllib.feature.InfoThCriterionFactory
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.Model
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.LogisticRegression
import scala.collection.mutable.Queue
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.regression.{LabeledPoint => OldLP}
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.sql.types.StructField
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.util.ModDiscretizerModel
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator



/**
 * @author sramirez
 */


object MainMLlibTest {
  var sqlContext: SQLContext = null
  var pathFile = "test_lung_s3.csv"
  var testFile = "test_lung_s3.csv"
  var order = -1 // Note: -1 means descending
  var nPartitions = 1
  var discretize = false
  var padded = 2
  var classLastIndex = false
  var clsLabel: String = null
  var inputLabel: String = "features"
  var firstHeader: Boolean = false
  var k: Int = 5
  var continuous: Boolean = false
  var nselect: Array[Int] = Array.emptyIntArray
  var seed = 12345678L
  var lowerFeatThreshold = 0.5
  var lowerDistanceThreshold = 0.8
  var numHashTables = 100
  var bucketWidth = 4
  var signatureSize = 5
  var mode = "test-lsh"
  var batchSize = 0.25f
  var estimationRatio = 1.0f
  var queryStep = 2
  var format = "csv"
  var sparseSpeedup = 0
  var predict: Boolean = false
  var savePreprocess: Boolean = false
  var sampling = 100
  var normalize: Boolean = false
  var repartition: Boolean = false
  var typePred: String = "all"
  
  
  // Case class for criteria/feature
  protected case class F(feat: Int, crit: Double)
  
  def main(args: Array[String]) {
    
    val initStartTime = System.nanoTime()
    
    val conf = new SparkConf().setAppName("CollisionFS Test")//.setMaster("local[*]").set("spark.driver.memory", "16g").set("spark.executor.memory", "16g")
    val sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)
    println("Usage: MLlibTest --train-file=\"hdfs://blabla\" --nselect=10 --npart=1 --continuous=false --k=5 --discretize=false --padded=2 --class-last=true --header=false")
        
    // Create a table of parameters (parsing)
    val params = args.map{ arg =>
        val param = arg.split("--|=").filter(_.size > 0)
        param.size match {
          case 2 =>  param(0) -> param(1)
          case _ =>  "" -> ""
        }
    }.toMap    
    
    pathFile = params.getOrElse("train-file", "src/test/resources/data/test_lung_s3.csv")
    testFile = params.getOrElse("test-file", "src/test/resources/data/test_lung_s3.csv")
    
    nPartitions = params.getOrElse("npart", "1").toInt
    discretize = params.getOrElse("disc", "false").toBoolean
    padded = params.getOrElse("padded", "0").toInt
    classLastIndex = params.getOrElse("class-last", "false").toBoolean
    firstHeader = params.getOrElse("header", "false").toBoolean
    k = params.getOrElse("k", "5").toInt
    nselect = params.getOrElse("nselect", "10").split(",").map { _.toInt }
    continuous = params.getOrElse("continuous", "true").toBoolean
    predict = params.getOrElse("predict", "false").toBoolean
    savePreprocess = params.getOrElse("savePreprocess", "false").toBoolean
    repartition = params.getOrElse("repartition", "false").toBoolean
    normalize = params.getOrElse("normalize", "false").toBoolean
    typePred = params.getOrElse("typePred", "all")
    lowerFeatThreshold = params.getOrElse("lowerFeatThreshold", "2.0").toFloat
    lowerDistanceThreshold = params.getOrElse("lowerDistanceThreshold", "0.8").toFloat
    numHashTables = params.getOrElse("numHashTables", "50").toInt
    bucketWidth = params.getOrElse("bucketWidth", "12").toInt
    signatureSize = params.getOrElse("signatureSize", "5").toInt
    batchSize = params.getOrElse("batchSize", "0.5f").toFloat
    estimationRatio = params.getOrElse("estimationRatio", "1.0f").toFloat
    queryStep = params.getOrElse("queryStep", "2").toInt    
    mode = params.getOrElse("mode", "final")
    format = params.getOrElse("format", "csv")
    sparseSpeedup = params.getOrElse("sparseSpeedup", "0").toInt
    sampling = params.getOrElse("sampling", "100").toInt
    
    println("Params used: " +  params.mkString("\n"))
    
    val rawDF = TestHelper.readData(sqlContext, pathFile, firstHeader, format)
    val partDF = if(repartition) rawDF.repartition(nPartitions).cache else rawDF.coalesce(nPartitions)
    val df = preProcess(partDF).select(clsLabel, inputLabel).cache()
    val nelems = df.count()
    println("# of examples readed and processed: " + nelems)
       
    
    if(mode == "test-lsh"){
      this.testLSHPerformance(df, nelems)
    } else if(mode == "final") {
      testFinalSelector(df, nelems)
    } else {
      doRELIEFComparison(df)
    }
    sc.stop()
  }
  
  
  def doRELIEFComparison(df: Dataset[_]) {
    
    val origRDD = initRDD(df.toDF(), allVectorsDense = true)
    val rdd = origRDD.map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
    }.cache //zipwithUniqueIndexs
    
    // Dataframe version
    val inputData = sqlContext.createDataFrame(origRDD, df.schema).cache()
    println("Schema: " + inputData.schema)
    
    val nf = rdd.first.features.size
    val nelems = rdd.count()
    
    val accMarginal = new VectorAccumulator(nf, false)
    // Then, register it into spark context:
    rdd.context.register(accMarginal, "marginal")
    val accJoint = new MatrixAccumulator(nf, nf, false)
    rdd.context.register(accJoint, "joint")
    val total = rdd.context.longAccumulator("total")
    println("# instances: " + rdd.count)
    println("# partitions: " + rdd.partitions.size)
    val knn = k
    val cont = continuous
    val lowerTh = 0.8f 
    val priorClass = rdd.map(_.label).countByValue().mapValues(_ / nelems.toFloat).map(identity)
    val bpriorClass = rdd.context.broadcast(priorClass)
    val nClasses = priorClass.size
    val lseed = this.seed
    
    val reliefRanking = rdd.mapPartitions { it =>
        
        val reliefWeights = breeze.linalg.DenseVector.fill(nf){0.0f}
        val marginal = breeze.linalg.DenseVector.zeros[Double](nf)
        val joint = breeze.linalg.DenseMatrix.zeros[Double](nf, nf)
        val ltotal = breeze.linalg.DenseVector.zeros[Long](nf)
        
        val elements = it.toArray
        val neighDist = breeze.linalg.DenseMatrix.fill(
              elements.size, elements.size){Double.MinValue}
        
        // BPQ replace always the lowest, so we have to change the order (very important not to modify -1)
        val ordering = Ordering[Double].on[(Double, Int)](-_._1) 
        val r = new scala.util.Random(lseed)
        // Data are assumed to be scaled to have 0 mean, and 1 std
        val vote = if(cont) (d: Double) => 1 - math.min(6.0, d) / 6.0 else (d: Double) => Double.MinPositiveValue
          
        (0 until elements.size).foreach{ id1 =>
          val e1 = elements(id1)
          var topk = Array.fill[BoundedPriorityQueue[(Double, Int)]](nClasses)(
                new BoundedPriorityQueue[(Double, Int)](knn)(ordering))
          val threshold = 6 * (1 - (lowerTh + r.nextFloat() * lowerTh))
          val condition = if(cont) (d: Double) => d <= threshold else 
                   (d: Double) => d == 0
                   
          (0 until elements.size).foreach{ id2 => 
            if(neighDist(id2, id1) < 0){
              if(id1 != id2) {
                // Compute collisions and distance
                val e2 = elements(id2)              
                var collisioned = Queue[Int]()
                neighDist(id1, id2) = 0 // Initialize the distance counter
                val pcounter = Array.fill(nf)(0.0d)
                e1.features.foreachActive{ (index, value) =>
                   val fdistance = math.abs(value - e2.features(index))
                   // The closer the distance, the greater the annotating likelihood.
                   if(condition(fdistance)){
                      val contribution = vote(fdistance)
                      marginal(index) += contribution
                      if(cont) pcounter(index) = contribution
                      val it = collisioned.iterator
                      while(it.hasNext){
                        val i2 = it.next
                        val jointVote = if(cont) (pcounter(i2) + pcounter(index)) / 2 else contribution
                        joint(i2, index) += jointVote
                      }                         
                      collisioned += index
                   }
                   neighDist(id1, id2) += math.pow(fdistance, 2)
                }
                neighDist(id1, id2) = math.sqrt(neighDist(id1, id2))  
                topk(elements(id2).label.toInt) += neighDist(id1, id2) -> id2
                total.add(1L)
              }
            } else {
              topk(elements(id2).label.toInt) += neighDist(id2, id1) -> id2              
            }                      
        }
        // RELIEF-F computations        
        e1.features.foreachActive{ case (index, value) =>
          val weight = (0 until nClasses).map { cls =>  
            val sum = topk(cls).map{ case(dist, id2) => math.abs(value - elements(id2).features(index)) }.sum
            if(cls != e1.label){
              val factor = bpriorClass.value.getOrElse(cls, 0.0f) / 
                (1 - bpriorClass.value.getOrElse(e1.label, 0.0f)) 
              sum.toFloat * factor / topk(cls).size
            } else {
              -sum.toFloat / topk(cls).size 
            }
          }.sum
          reliefWeights(index) += weight       
        }
      }
      // update accumulated matrices  
      accMarginal.add(marginal)
      accJoint.add(joint)
      
      reliefWeights.iterator      
    }.reduceByKey(_ + _).cache
    
    println("Relief ranking: " + reliefRanking.sortBy(-_._2).collect.mkString("\n"))
    println("Number of collisions by feature: " + total.value)
    
    val maxRelief = reliefRanking.values.max()
    val minRelief = reliefRanking.values.min()
    val normalizedRelief = reliefRanking.mapValues(score => ((score - minRelief) / (maxRelief - minRelief)).toFloat).collect
    
    val factor = if(cont) 1.0 else Double.MinPositiveValue
    val marginal = accMarginal.value.toDenseVector.mapPairs{ case(index, value) =>
      value /  (total.value * factor)
    }  
    val joint = accJoint.value.toDenseMatrix.mapPairs{ case(index, value) => 
      value / (total.value.longValue() * factor)
    }  
    
    // Compute mutual information using collisions with and without class
    val redundancyMatrix = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    joint.activeIterator.foreach { case((i1,i2), value) =>
      if(i1 < i2) {
        val red = (value * log2(value / (marginal(i1) * marginal(i2)))).toFloat              
        redundancyMatrix(i1, i2) = red
        redundancyMatrix(i2, i1) = red        
      }        
    }
    
    val maxRed = breeze.linalg.max(redundancyMatrix)
    val minRed = breeze.linalg.min(redundancyMatrix)
    
    val normRedundancyMatrix = redundancyMatrix.mapValues{ value => ((value - minRed) / (maxRed - minRed)).toFloat }    
    val (reliefColl, relief) = selectFeatures(nf, normalizedRelief, normRedundancyMatrix)    
    val reliefCollModel = new InfoThSelectorModel("", new org.apache.spark.mllib.feature.InfoThSelectorModel(
        selectedFeatures = reliefColl.map { case F(feat, rel) => feat }.sorted.toArray))
          .setOutputCol("selectedFeatures")
          .setFeaturesCol(inputLabel) // this must be a feature vector
          .setLabelCol(clsLabel)
          
    val reliefModel = new InfoThSelectorModel("", new org.apache.spark.mllib.feature.InfoThSelectorModel(
        selectedFeatures = relief.map { case F(feat, rel) => feat }.sorted.toArray))
          .setOutputCol("selectedFeatures")
          .setFeaturesCol(inputLabel) // this must be a feature vector
          .setLabelCol(clsLabel)
    
    // Print best features according to the RELIEF-F measure
    val outRC = reliefColl.map { case F(feat, rel) => (feat + 1) + "\t" + "%.4f".format(rel) }.mkString("\n")
    val outR = relief.map { case F(feat, rel) => (feat + 1) + "\t" + "%.4f".format(rel) }.mkString("\n")
    println("\n*** RELIEF + Collisions selected features ***\nFeature\tScore\n" + outRC)
    println("\n*** RELIEF selected features ***\nFeature\tScore\n" + outR)
    
    var mrmrAcc = 0.0f; var mrmrAccDT = 0.0f; var mrmrAccLR = 0.0f; var selectedMRMR = new String();
    if(typePred == "all" || typePred == "onlymrmr"){      
      val mRMRmodel = fitMRMR(inputData, 100)
      println("\n*** Selected by mRMR: " + mRMRmodel.selectedFeatures.map(_ + 1).mkString(","))
      mrmrAccDT = kCVPerformance(mRMRmodel.transform(inputData), "dt")
      mrmrAccLR = kCVPerformance(mRMRmodel.transform(inputData), "lr")
      selectedMRMR = mRMRmodel.selectedFeatures.map(_ + 1).mkString(",")
    }
    
    var relCAccDT = 0.0; var relAccDT = 0.0; var accDT = 0.0; 
    var relCAccLR = 0.0; var relAccLR = 0.0; var accLR = 0.0;
    if(typePred == "all" || typePred == "onlyrelief"){      
      val reducedRC = reliefCollModel.transform(inputData).cache()
      val reducedR = reliefModel.transform(inputData).cache()
      
      val relCAccDT = kCVPerformance(reducedRC, "dt")   
      val relAccDT = kCVPerformance(reducedR, "dt")   
      val accDT = kCVPerformance(inputData, "dt")   
      val relCAccLR = kCVPerformance(reducedRC, "lr") 
      val relAccLR = kCVPerformance(reducedR, "lr") 
      val accLR = kCVPerformance(inputData, "lr")
    }

    println("Train accuracy for mRMR (Decision Tree) = " + mrmrAccDT)
    println("Train accuracy for ReliefColl (Decision Tree) = " + relCAccDT)
    println("Train accuracy for Relief (Decision Tree) = " + relAccDT)
    println("Baseline train accuracy (Decision Tree) = " + accDT)
    println("Train accuracy for mRMR (LR) = " + mrmrAccLR)
    println("Train accuracy for ReliefColl (LR) = " + relCAccLR)
    println("Train accuracy for Relief (LR) = " + relAccLR)
    println("Baseline train accuracy (LR) = " + accLR)
    
    println("\n*** Selected by mRMR: " + selectedMRMR)
    println("\n*** RELIEF + Collisions selected features ***\nFeature\tScore\n" + outRC)
    println("\n*** RELIEF selected features ***\nFeature\tScore\n" + outR)
  }
  
  def holdOutPerformance(df: Dataset[Row], test: Dataset[Row], classifier: String) = {
    var labelCol = clsLabel
    var inputCol = if(df.schema.fieldNames.exists { _ == "selectedFeatures" }) "selectedFeatures" else inputLabel
    val sql = df.sqlContext
    
    val estimator = if(classifier == "nb") {
       new NaiveBayes()
        .setFeaturesCol(inputCol)
        .setLabelCol(labelCol)    
     
    } else if(classifier == "dt") {
      // all features have been indexed previously in the preProcess method (label included).
      new DecisionTreeClassifier()
        .setFeaturesCol(inputCol)
        .setLabelCol(labelCol)  
      
    } else {
      new LinearSVC()
        .setFeaturesCol(inputCol)
        .setLabelCol(labelCol) 
    }
    
    val evacc = new MulticlassClassificationEvaluator()
        .setLabelCol(labelCol)
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

    val evf1 = new MulticlassClassificationEvaluator()
        .setLabelCol(labelCol)
        .setPredictionCol("prediction")
        .setMetricName("f1")
        
    val evrecall = new MulticlassClassificationEvaluator()
        .setLabelCol(labelCol)
        .setPredictionCol("prediction")
        .setMetricName("weightedRecall")
        
    val evPred = new MulticlassClassificationEvaluator()
        .setLabelCol(labelCol)
        .setPredictionCol("prediction")
        .setMetricName("weightedPrecision")
        
        
        
    //K-folding operation starting
    //for each fold you have multiple models created cfm. the paramgrid
    val input = df.select(clsLabel, inputCol); val state = input.storageLevel
    val persistedInput = if(state == StorageLevel.NONE) input.cache() else input // avoid duplicates
    val model = estimator.fit(input)
    val predictions = model.transform(test)
    val acc = evacc.evaluate(predictions)
    val recall = evrecall.evaluate(predictions)
    val prec = evPred.evaluate(predictions)
    val f1 = evf1.evaluate(predictions)
    
    if(state == StorageLevel.NONE) 
      input.unpersist()
    (acc, recall, prec, f1)
  }
  
  def kCVPerformance(df: Dataset[Row], classifier: String) = {
    
    var labelCol = clsLabel
    var inputCol = if(df.schema.fieldNames.exists { _ == "selectedFeatures" }) "selectedFeatures" else inputLabel
    val splits = MLUtils.kFold(df.rdd, 10, seed)
    val sql = df.sqlContext
    
    val estimator = if(classifier == "nb") {
       new NaiveBayes()
        .setFeaturesCol(inputCol)
        .setLabelCol(labelCol)    
     
    } else if(classifier == "dt") {
      val labelIndexer = new StringIndexer()
          .setInputCol(labelCol)
          .setOutputCol("indexedLabel")
          .fit(df)
      labelCol = "indexedLabel"    
      // Automatically identify categorical features, and index them.
      val featureIndexer = new VectorIndexer()
        .setInputCol(inputCol)
        .setOutputCol("indexedFeatures")
        .setMaxCategories(15) // features with > 4 distinct values are treated as continuous.
        .fit(df)
        
      inputCol = "indexedFeatures"
      
      val dt = new DecisionTreeClassifier()
        .setFeaturesCol(inputCol)
        .setLabelCol(labelCol)  
        
      // Convert indexed labels back to original labels.
      val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictedLabel")
        .setLabels(labelIndexer.labels)
        
      new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
      //new Pipeline().setStages(Array(dt))

    } else {
      new LogisticRegression()
        .setFeaturesCol(inputCol)
        .setLabelCol(labelCol) 
    }
    
    val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol(labelCol)
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
        
        
    //K-folding operation starting
    //for each fold you have multiple models created cfm. the paramgrid
    val sum = splits.map { case (training, validation) =>
      val trainingDataset = sql.createDataFrame(training, df.schema).cache()
      val validationDataset = sql.createDataFrame(validation, df.schema).cache()
      
      val model = estimator.fit(trainingDataset)
      trainingDataset.unpersist()
      val acc = evaluator.evaluate(model.transform(validationDataset))
      if(!acc.isNaN()){
        acc
      } else {
        0.0d
      }
    }.sum
    sum.toFloat / splits.size
  }
  
  
  
  def selectFeatures(nfeatures: Int, reliefRanking: Array[(Int, Float)],
      redundancyMatrix: breeze.linalg.DenseMatrix[Float]) = {
    
    val attAlive = Array.fill(nfeatures)(true)
    // Initialize all (except the class) criteria with the relevance values
    val criterionFactory = new InfoThCriterionFactory("mrmr")
    val pool = Array.fill[InfoThCriterion](nfeatures) {
      val crit = criterionFactory.getCriterion.init(Float.NegativeInfinity)
      crit.setValid(false)
    }
    
    reliefRanking.foreach {
      case (x, mi) =>
        pool(x) = criterionFactory.getCriterion.init(mi.toFloat)
    }

    // Get the maximum and initialize the set of selected features with it
    val (max, mid) = pool.zipWithIndex.maxBy(_._1.relevance)
    var selected = Seq(F(mid, max.score))
    pool(mid).setValid(false)
    
    var moreFeat = true
    // Iterative process for redundancy and conditional redundancy
    while (selected.size < nselect.max && moreFeat) {

      attAlive(selected.head.feat) = false
      val redundancies = redundancyMatrix(::, selected.head.feat)
              .toArray
              .zipWithIndex
              .filter(c => attAlive(c._2))

      // Update criteria with the new redundancy values      
      redundancies.par.foreach({
        case (mi, k) =>            
          pool(k).update(mi.toFloat, 0.0f)
      })
      
      // select the best feature and remove from the whole set of features
      val (max, maxi) = pool.zipWithIndex.filter(_._1.valid).sortBy(c => (-c._1.score, c._2)).head
      
      if (maxi != -1) {
        selected = F(maxi, max.score) +: selected
        pool(maxi).setValid(false)
      } else {
        moreFeat = false
      }
    }
    val reliefNoColl = reliefRanking.sortBy(r => (-r._2, r._1))
        .slice(0, nselect.max).map{ case(id, score) => F(id, score)}.toSeq
    (selected.reverse, reliefNoColl)  
  }
  
  def preProcess(df: DataFrame) = {
    val other = if(classLastIndex) df.columns.dropRight(1) else df.columns.drop(1)
    clsLabel = if(classLastIndex) df.columns.last else df.columns.head
    
    // Index categorical values
    val stringTypes = df.dtypes.filter(_._2 == "StringType").map(_._1)
    val tmpNames = df.dtypes.map{ case(name, typ) => if(typ == "StringType") name + "-indexed" else name}
    clsLabel = if(classLastIndex) tmpNames.last else tmpNames.head
    val newNames = if(classLastIndex) tmpNames.dropRight(1) else tmpNames.drop(1)
    val indexers = stringTypes.map{ name =>
        new StringIndexer()
          .setInputCol(name)
          .setOutputCol(name + "-indexed")
    }

    val pipeline = new Pipeline().setStages(indexers)
    val typedDF = pipeline.fit(df).transform(df).drop(stringTypes: _*)
    println("Indexed Schema: " + typedDF.schema)
    
    // Clean Label Column
    val cleanedDF = TestHelper.cleanLabelCol(typedDF, clsLabel)
    clsLabel = clsLabel + TestHelper.INDEX_SUFFIX
    println("clslabel: " + clsLabel)
    
    // Assemble all input features
    var processedDF = if(newNames.size > 1){
      val featureAssembler = new VectorAssembler()
        .setInputCols(newNames)
        .setOutputCol(inputLabel)
      featureAssembler.transform(cleanedDF).select(clsLabel, inputLabel)
    } else {
      cleanedDF.select(clsLabel, inputLabel)
    }
    
    // If format is not csv and most of instances are sparse, then all are transformed to sparse
    val majoritySparse = if(format == "csv") false else processedDF.rdd.map{case Row(cls: Double, features: Vector) => 
      features.isInstanceOf[SparseVector]}.countByValue().max._1 
    val standarizeTypeUDF = udf((feat: Vector) => if(majoritySparse) feat.toSparse else feat.toDense)
    processedDF = processedDF.withColumn(inputLabel, standarizeTypeUDF(col(inputLabel)))
    println("clsLabel: " + clsLabel)
    println("Columns: " + processedDF.columns.mkString(","))
    println("Schema: " + processedDF.schema.toString)
      
    if(discretize) {
      // Continuous data from LIBSVM has to be discretized since
      // ML does not allow fair normalization of sparse vectors
      val discretizer = new MDLPDiscretizer()
        .setMaxBins(15)
        .setMaxByPart(10000)
        .setInputCol(inputLabel)
        .setLabelCol(clsLabel)
        .setOutputCol("disc-" + inputLabel)
        .setApproximate(true)
        
      val model = discretizer.fit(processedDF)
      val rddModel = new ModDiscretizerModel(model.splits)
      val inputRDD = processedDF.select(clsLabel, inputLabel).rdd.map{
        case Row(l: Double, v: Vector) => OldLP(l, OldVectors.fromML(v)) }
      val discRDD = rddModel.transformRDD(inputRDD)
      if(savePreprocess) 
        MLUtils.saveAsLibSVMFile(discRDD, pathFile + ".disc")
      inputLabel = "disc-" + inputLabel
      val schema = new StructType()
            .add(StructField(inputLabel, new VectorUDT(), true))
            .add(StructField(clsLabel, DoubleType, true))
      val discDF = sqlContext.createDataFrame(discRDD.map{ lp => Row(lp.features.asML, lp.label)}, schema).cache()      
      processedDF = discDF      
      continuous = false
    } else if(normalize) {
      val scaler = new StandardScaler()
        .setInputCol(inputLabel)
        .setOutputCol("norm-" + inputLabel)
        .setWithStd(true)
        .setWithMean(false) // avoids problems with sparse data
      
      val smodel = scaler.fit(processedDF)
      processedDF = smodel.transform(processedDF)
      inputLabel = "norm-" + inputLabel
    }
    if(savePreprocess) {
      if(format == "csv"){
        processedDF.select(clsLabel, inputLabel).rdd
          .map{case Row(label: Double, features: Vector) => features.toArray.mkString(",") + "," + label}
          .saveAsTextFile(pathFile + ".disc")
      } else if (format == "libsvm" && !discretize) {
        val output = processedDF.select(clsLabel, inputLabel).rdd
              .map{case Row(label: Double, features: Vector) => OldLP(label, OldVectors.fromML(features))}
        MLUtils.saveAsLibSVMFile(output, pathFile + ".disc")
      }
    }
    processedDF
  }
   
  def initRDD(df: DataFrame, allVectorsDense: Boolean) = {
    val pad = padded // Needed to avoid task not serializable exception (raised by the class by itself)
    df.rdd.map {
      case Row(label: Double, features: Vector) =>
        val standardv = if(allVectorsDense){
          Vectors.dense(features.toArray.map(_ + pad))
        } else {
            val sparseVec = features.toSparse
            val newValues: Array[Double] = sparseVec.values.map(_ + pad)
            Vectors.sparse(sparseVec.size, sparseVec.indices, newValues)
        }        
        Row.fromSeq(Seq(label, standardv))
    }
  }

  
  def fitMRMR(df: Dataset[_], nSelect: Int, hardcodedLoad: Boolean = false) = {
    
    val dataname = pathFile.split("/").last.split(Array('-','.','_')).head
    var model: InfoThSelectorModel = null
    
    if(hardcodedLoad){
      val selected = dataname match {
        case "ECBDL14" => Array(1,62,17,433,151,176,153,2,10,496,402,3,547,33,316,587,251,364,124,71,476,296,211,117,338,133,431,22,271,116,
            525,224,384,604,120,471,155,276,7,129,552,254,114,451,165,511,354,544,128,484,378,187,616,130,514,100,16,212,26,44,161,584,53,122,
            307,327,564,123,396,232,607,516,32,196,166,8,428,339,112,452,171,290,368,113,80,492,167,185,89,125,624,401,18,336,557,27,487,236,157,36)
        case "kddb" => Array(3284,2281,4189,3893,2417,958,133,1523,1514,2515,11376,2279,6079,87265,6655,755,771,152783,3770,2256,4006,7549,2538,6090,129653,
            304,49255,4199,3407,4064,665,254774,19777,4924,4039,156,2415,6036,18039,16390,4187,11375,6298,28710,951,6077,216903,32464,3285,6696,3913,4936,7732,
            2285,1547,125422,24883,254773,2254,797,3427,49333,3302,895,3287,131,216912,4056,3764,16387,6080,1524,6888,4955,2416,1511,4209,158,12097,11612,3310,
            7558,770,87267,4001,39481,3328,4188,19781,2255,4937,6697,3319,2282,185512,957,16388,6078,6027,4126)
        case "url" => Array(28,17,6,48,155,270,47,4,155162,70,263,155165,912,216,22,51,155214,355,24,267,5,110,194,909,217,2281,266,59,2401,21,363,18,913,2282,
            53,3521,54,155168,362,139,2284,90,30,215,910,368,19,2283,155172,356,1736,906,1306,185,3522,271,61,25,11,8197,352,155169,372,274,8200,50,23,1309,39,905,
            2510,94,10728,1560,2285,8199,1310,705,272,155173,8099,908,367,2288,4582,8198,353,1866,696,915,161,298,1629,275,141,74,4583,914,2286,1032)
        case "epsilon" => Array(819,1867,650,2,1698,4,1390,1520,101,5,937,316,995,6,148,8,9,1795,469,10,11,12,344,13,1087,1279,14,15,16,756,1401,17,18,21,1052,22,23,790,
            24,918,25,26,381,27,28,29,830,691,30,31,32,1563,218,33,34,35,36,37,39,1923,40,42,44,45,831,46,47,48,49,696,51,52,53,54,55,1937,56,710,1338,57,58,59,61,62,
            485,64,543,66,67,68,70,71,72,73,855,74,75,146,76,77)
      }
      val oldITSelector = new org.apache.spark.mllib.feature.InfoThSelectorModel(selected.slice(0, nSelect).map(_ - 1).sorted)
      model = new InfoThSelectorModel("mRMR-" + dataname + "-" + nSelect, oldITSelector).setOutputCol("selectedFeatures")
    } else {
      val modelPath = "MRMR-model-" + dataname + "-" + numHashTables + "-" + bucketWidth + "-" +
        signatureSize + "-" + k + "-" + estimationRatio + "-" + batchSize + "-" + lowerFeatThreshold
      val selector = new InfoThSelector()
          .setSelectCriterion("mrmr")
          .setNPartitions(nPartitions)
          .setNumTopFeatures(nselect.max)
          .setFeaturesCol(inputLabel) // this must be a feature vector
          .setLabelCol(clsLabel)
          .setOutputCol("selectedFeatures")
      try {
          model = InfoThSelectorModel.load(modelPath)
      } catch {
        case t: Throwable => {
          t.printStackTrace() // TODO: handle error
          model = selector.fit(df)
          model.save(modelPath)
        }
      }
    }
    
    model
  }
  
  def testLSHPerformance(df: Dataset[_], nelems: Long) {
    
    val nFeat = df.select(inputLabel).head().getAs[Vector](0).size
    val kfold = 3
    
    val brp = new BucketedRandomLSH()
      .setNumHashTables(numHashTables)
      .setInputCol(inputLabel)
      .setOutputCol("hashCol")
      .setBucketLength(bucketWidth)
      .setSignatureSize(signatureSize)
      .setSparseSpeedup(sparseSpeedup)
      .setSeed(seed)
   var model = brp.fit(df)
      
      // Sample only some elements to test
    val samplingRate = sampling * kfold / nelems.toDouble // k-fold cross validation, k = 10
    val keys = df.select(inputLabel).sample(false, samplingRate, seed).collect()
    println("Number of examples used in estimation: " + keys.length)
    
    var sumer = 0.0; var sump = 0.0; var sumr = 0.0; var red = 0L; var sumtime = 0.0; 
    var sumMax = 0.0; var maxd = 0.0; var cont = 0
    keys.foreach { case Row(key: Vector) =>  
      val (errorRatio, precision, recall, redundancy, time) = LSHTest.calculateApproxNearestNeighbors(
          model, df, key, k, "multi", "distCol", nelems) 
      sump += precision; sumr += recall; sumtime += time; sumer += errorRatio
      cont += 1
      if(cont % kfold == 0)
        model = brp.fit(df)
      println("# instances completed: " + cont)
    }
    
    println("Average precision: " + sump / keys.size)
    println("Average recall: " + sumr / keys.size)
    println("Average selectivity: " + red / keys.size)
    println("Average runtime (in s): " + sumtime / keys.size)
    println("Number of hash tables: " + numHashTables)
    println("Signature size: " + signatureSize)
    println("Bucket width: " + bucketWidth)
    println("Average error ratio (distance): " + sumer / keys.size)
    println("Average maximum distance: " + sumMax / keys.size)
    println("Total Maximum distance: " + maxd)
    
  }
  
  def testFinalSelector(df: Dataset[Row], nElems: Long) {
    
    val first = df.head().getAs[Vector](inputLabel)
    val nFeat = first.size
    val sparse = first.isInstanceOf[SparseVector]
    println("Total features: " + nFeat)
    
    val selector = new ReliefFRSelector()
      .setInputCol(inputLabel)
      .setOutputCol("selectedFeatures")
      .setLabelCol(clsLabel)
      .setNumHashTables(numHashTables)
      .setBucketLength(bucketWidth)
      .setSignatureSize(signatureSize)
      .setSparseSpeedup(sparseSpeedup)
      .setSeed(seed)
      .setNumNeighbors(k)
      .setNumTopFeatures(nselect.max)
      .setEstimationRatio(estimationRatio)
      .setBatchSize(batchSize)
      .setLowerFeatureThreshold(lowerFeatThreshold)
      .setLowerDistanceThreshold(lowerDistanceThreshold)
      .setQueryStep(queryStep)
      .setDiscreteData(!continuous)
    
    val now = System.currentTimeMillis
    val dataname = pathFile.split("/").last.split(Array('-','.','_')).head
    val modelPath = "RELIEF-model-" + dataname + "-" + numHashTables + bucketWidth +
        signatureSize + k + estimationRatio + batchSize + lowerFeatThreshold
    var model: ReliefFRSelectorModel = null
    try {
        model = ReliefFRSelectorModel.load(modelPath)
    } catch {
      case t: Throwable => {
        t.printStackTrace() // TODO: handle error
        model = selector.fit(df)
        model.save(modelPath)
      }
    }
    val runtime = (System.currentTimeMillis - now) / 1000
    println("RELIEF-F model training time (seconds) = " + runtime)
    
    val outRC = model.setRedundancyRemoval(true).setReducedSubset(nselect.max).getSelectedFeatures().mkString("\n")
    println("\n*** RELIEF + Collisions selected features ***\nFeature\tScore\n" + outRC)
    val outR = model.setRedundancyRemoval(false).setReducedSubset(nselect.max).getSelectedFeatures().mkString("\n")
    println("\n*** RELIEF selected features ***\nFeature\tScore\n" + outR)
    
    if(predict) {
      
      inputLabel = "features"
      val rawTestDF = TestHelper.readData(sqlContext, testFile, firstHeader, format)
      val testDF = preProcess(rawTestDF).select(clsLabel, inputLabel).cache()
      
      // Print best features according to the RELIEF-F measure
      nselect.reverse.foreach{ nfeat => 
        
        var mrmrAccDT = ""; var mrmrAccLR = ""; var selectedMRMR = new String();
        if(typePred == "all" || typePred == "onlymrmr"){        
          val now = System.currentTimeMillis
          val mRMRmodel = fitMRMR(df, nfeat, hardcodedLoad = true)
          val runtime = (System.currentTimeMillis - now) / 1000
          println("mRMR model training time (in s) = " + runtime)
          selectedMRMR = mRMRmodel.selectedFeatures.map(_ + 1).mkString(",")
          println("\n*** Selected by mRMR: " + selectedMRMR)
          val trReducedMRMR = mRMRmodel.transform(df)
          val reducedMRMR = mRMRmodel.transform(testDF)
          mrmrAccDT = holdOutPerformance(trReducedMRMR, reducedMRMR, "dt").toString()
          mrmrAccLR = holdOutPerformance(trReducedMRMR, reducedMRMR, "svc").toString()          
        }
        
        var relCAccDT = ""; var relAccDT = ""
        var relCAccLR = ""; var relAccLR = ""
        if(typePred == "all" || typePred == "onlyrelief"){      
          val partialModel = model.setReducedSubset(nfeat)
          val outRC = partialModel.setRedundancyRemoval(true).getSelectedFeatures().mkString("\n")
          val outR = partialModel.setRedundancyRemoval(false).getSelectedFeatures().mkString("\n")
          println("\n*** RELIEF + Collisions selected features ***\nFeature\tScore\n" + outRC)
          println("\n*** RELIEF selected features ***\nFeature\tScore\n" + outR)
            
          val reducedRC = partialModel.setRedundancyRemoval(true).transform(df)
          val tReducedRC = partialModel.setRedundancyRemoval(true).transform(testDF)
          relCAccDT = holdOutPerformance(reducedRC, tReducedRC, "dt").toString() 
          relCAccLR = holdOutPerformance(reducedRC, tReducedRC, "svc").toString() 
          
          val reducedR = partialModel.setRedundancyRemoval(false).transform(df)
          val tReducedR = partialModel.setRedundancyRemoval(false).transform(testDF)
          relAccDT = holdOutPerformance(reducedR, tReducedR, "dt").toString()
          relAccLR = holdOutPerformance(reducedR, tReducedR, "svc").toString()
        }        
        
        val accDT = if(!sparse) holdOutPerformance(df, testDF, "dt").toString() else "0.0"
        val accLR = holdOutPerformance(df, testDF, "svc").toString() 
    
        println("Test (acc, recall, prec, f1) for mRMR-" + nfeat + "-DT = " + mrmrAccDT)
        println("Test (acc, recall, prec, f1) for Reliefc-" + nfeat + "DT = " + relCAccDT)
        println("Test (acc, recall, prec, f1) for Relief-" + nfeat + "DT = " + relAccDT)
        println("Test (acc, recall, prec, f1) for baseline-0-DT = " + accDT)
        println("Test (acc, recall, prec, f1) for mRMR-" + nfeat + "-LR = " + mrmrAccLR)
        println("Test (acc, recall, prec, f1) for Reliefc-" + nfeat + "LR = " + relCAccLR)
        println("Test (acc, recall, prec, f1) for Relief-" + nfeat + "LR = " + relAccLR)
        println("Test (acc, recall, prec, f1) for baseline-0-LR = " + accLR)  
      
      }
    }
    println("\n*** Modeling runtime for FS (seconds) = " + runtime)    

  }
  
  
  def log2(x: Double) = { math.log(x) / math.log(2) }
}
