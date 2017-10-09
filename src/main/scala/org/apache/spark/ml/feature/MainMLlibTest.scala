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


/**
 * @author sramirez
 */


object MainMLlibTest {
  var sqlContext: SQLContext = null
  var pathFile = "test_lung_s3.csv"
  var order = -1 // Note: -1 means descending
  var nPartitions = 1
  var nTop = 10
  var discretize = false
  var padded = 2
  var classLastIndex = false
  var clsLabel: String = null
  var inputLabel: String = "features"
  var firstHeader: Boolean = false
  var k: Int = 5
  var continuous: Boolean = false
  var nselect: Int = 10
  var seed = 12345678L
  
  var mrmr: Boolean = false
  
  
  // Case class for criteria/feature
  protected case class F(feat: Int, crit: Double)
  
  def main(args: Array[String]) {
    
    val initStartTime = System.nanoTime()
    
    val conf = new SparkConf().setAppName("CollisionFS Test").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)

    println("Usage: MLlibTest --train-file=\"hdfs://blabla\" --nselect=10 --npart=1 --continuous=false --k=5 --ntop=10 --discretize=false --padded=2 --class-last=true --header=false")
        
    // Create a table of parameters (parsing)
    val params = args.map{ arg =>
        val param = arg.split("--|=").filter(_.size > 0)
        param.size match {
          case 2 =>  param(0) -> param(1)
          case _ =>  "" -> ""
        }
    }.toMap    
    
    pathFile = params.getOrElse("train-file", "src/test/resources/data/test_lung_s3.csv")
    nPartitions = params.getOrElse("npart", "1").toInt
    nTop = params.getOrElse("ntop", "10").toInt
    discretize = params.getOrElse("disc", "false").toBoolean
    padded = params.getOrElse("padded", "2").toInt
    classLastIndex = params.getOrElse("class-last", "false").toBoolean
    firstHeader = params.getOrElse("header", "false").toBoolean
    k = params.getOrElse("k", "10").toInt
    nselect = params.getOrElse("nselect", "10").toInt
    continuous = params.getOrElse("continuous", "true").toBoolean
    mrmr = params.getOrElse("mrmr", "false").toBoolean
    
    
    
    
    println("Params used: " +  params.mkString("\n"))
    
    doRELIEFComparison()
    //doComparison()
  }
  
  
  def doRELIEFComparison() {
    val rawDF = TestHelper.readCSVData(sqlContext, pathFile, firstHeader)
    val df = preProcess(rawDF).select(clsLabel, inputLabel)
    val allVectorsDense = true
 
    df.show
    println("df: " + df.first().toString())
    
    val origRDD = initRDD(df, allVectorsDense)
    val rdd = origRDD.map {
      case Row(label: Double, features: Vector) =>
        LabeledPoint(label, features)
    }.repartition(nPartitions).cache //zipwithUniqueIndexs
    
    println("Discretized: " + rdd.map(_.features).take(100).mkString("\n"))
    println("rdd: " + rdd.first().toString())   
    
    //Dataframe version
    val inputData = sqlContext.createDataFrame(origRDD, df.schema).cache()
    println("Schema: " + df.schema)
    
    //val elements = rdd.collect
    val nf = rdd.first.features.size
    val nelems = rdd.count()
    
    val accMarginal = new VectorAccumulator(nf)
    // Then, register it into spark context:
    rdd.context.register(accMarginal, "marginal")
    val accJoint = new MatrixAccumulator(nf, nf)
    rdd.context.register(accJoint, "joint")
    val total = rdd.context.longAccumulator("total")
    println("# instances: " + rdd.count)
    println("# partitions: " + rdd.partitions.size)
    val knn = k
    val cont = continuous
    val priorClass = rdd.map(_.label).countByValue().mapValues(_ / nelems.toFloat).map(identity)
    val bpriorClass = rdd.context.broadcast(priorClass)
    val nClasses = priorClass.size
    
    val reliefRanking = rdd.mapPartitions { it =>
        
        val reliefWeights = breeze.linalg.DenseVector.fill(nf){0.0f}
        val marginal = breeze.linalg.DenseVector.zeros[Double](nf)
        val joint = breeze.linalg.DenseMatrix.zeros[Double](nf, nf)
        
        val elements = it.toArray
        val neighDist = breeze.linalg.DenseMatrix.fill(
              elements.size, elements.size){Double.MinValue}
        
        // BPQ replace always the lowest, so we have to change the order (very important not to modify -1)
        val ordering = Ordering[Double].on[(Double, Int)](-_._1) 
        val r = new scala.util.Random
        val condition = if(cont) (d: Double) => d > 0.5 + r.nextFloat() * 0.5 else (d: Double) => d == 0
        val vote = if(cont) (d: Double) => d else (d: Double) => Double.MinPositiveValue
          
        (0 until elements.size).foreach{ id1 =>
          val e1 = elements(id1)
          var topk = Array.fill[BoundedPriorityQueue[(Double, Int)]](nClasses)(
                new BoundedPriorityQueue[(Double, Int)](knn)(ordering))
                
          (0 until elements.size).foreach{ id2 => 
            
            if(neighDist(id2, id1) < 0){
              if(id1 != id2) {
                // Compute collisions and distance
                val e2 = elements(id2)              
                var collisioned = Queue[Int]()
                neighDist(id1, id2) = 0 // Init the distance counter
                e1.features.foreachActive{ (index, value) =>
                   val norm = math.abs(value - e2.features(index))
                   // We annotate the collision. 
                   // The closer the distance, the greater the annotating likelihood.
                   if(condition(norm)){
                      marginal(index) += vote(norm)
                      val it = collisioned.iterator
                      while(it.hasNext)
                         joint(it.next, index) += vote(norm)
                      collisioned += index
                   }
                   neighDist(id1, id2) += math.pow(norm, 2)
                }
                neighDist(id1, id2) = math.sqrt(neighDist(id1, id2))  
                topk(elements(id2).label.toInt) += neighDist(id1, id2) -> id2
                total.add(1L) // use to compute likelihoods (denom)
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
                ((1 - bpriorClass.value.getOrElse(e1.label, 0.0f)) * topk(cls).size)
              sum.toFloat * factor 
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
    val avgRelief = reliefRanking.values.mean()
    val stdRelief = reliefRanking.values.stdev()
    val normalizedRelief = reliefRanking.mapValues(score => ((score - avgRelief) / stdRelief).toFloat).collect()
    
    val marginal = accMarginal.value.mapValues(_ / (total.value * Double.MinPositiveValue))  
    val joint = accJoint.value.toDenseMatrix.mapValues(_ / (total.value * Double.MinPositiveValue))
    
    // Compute mutual information using collisions with and without class
    val redundancyMatrix = breeze.linalg.DenseMatrix.zeros[Float](nf, nf)
    joint.activeIterator.foreach { case((i1,i2), value) =>
      if(i1 < i2){
        val red = (value * log2(value / (marginal(i1) * marginal(i2)))).toFloat              
        redundancyMatrix(i1, i2) = red
        redundancyMatrix(i2, i1) = red        
      }        
    }
    
    import breeze.stats._ 
    val stats = meanAndVariance(redundancyMatrix)
    val normRedundancyMatrix = redundancyMatrix.mapValues{ value => ((value - stats.mean) / stats.stdDev).toFloat }    
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
    val mRMRmodel = fitMRMR(inputData)
    
    println("\n*** Selected by mRMR: " + mRMRmodel.selectedFeatures.map(_ + 1).mkString(","))
    println("\n*** RELIEF + Collisions selected features ***\nFeature\tScore\n" + outRC)
    println("\n*** RELIEF selected features ***\nFeature\tScore\n" + outR)
    
    val mrmrAcc = kCVPerformance(inputData, mRMRmodel, "nb")
    val relCAcc = kCVPerformance(inputData, reliefCollModel, "nb")   
    val relAcc = kCVPerformance(inputData, reliefModel, "nb")   
    val acc = kCVPerformance(inputData, null, "nb")   
    val mrmrAccDT = kCVPerformance(inputData, mRMRmodel, "dt")
    val relCAccDT = kCVPerformance(inputData, reliefCollModel, "dt")   
    val relAccDT = kCVPerformance(inputData, reliefModel, "dt")   
    val accDT = kCVPerformance(inputData, null, "dt")   
    val mrmrAccLR = kCVPerformance(inputData, mRMRmodel, "lr")
    val relCAccLR = kCVPerformance(inputData, reliefCollModel, "lr") 
    val relAccLR = kCVPerformance(inputData, reliefModel, "lr") 
    val accLR = kCVPerformance(inputData, null, "lr")

    println("Train accuracy for mRMR (Naive Bayes) = " + mrmrAcc)
    println("Train accuracy for Relief (Naive Bayes) = " + relAcc)
    println("Train accuracy for ReliefColl (Naive Bayes) = " + relCAcc)
    println("Baseline train accuracy (Naive Bayes) = " + acc)
    println("Train accuracy for mRMR (Decision Tree) = " + mrmrAccDT)
    println("Train accuracy for ReliefColl (Decision Tree) = " + relCAccDT)
    println("Train accuracy for Relief (Decision Tree) = " + relAccDT)
    println("Baseline train accuracy (Decision Tree) = " + accDT)
    println("Train accuracy for mRMR (LR) = " + mrmrAccLR)
    println("Train accuracy for ReliefColl (LR) = " + relCAccLR)
    println("Train accuracy for Relief (LR) = " + relAccLR)
    println("Baseline train accuracy (LR) = " + accLR)
    
    println("\n*** Selected by mRMR: " + mRMRmodel.selectedFeatures.map(_ + 1).mkString(","))
    println("\n*** RELIEF + Collisions selected features ***\nFeature\tScore\n" + outRC)
    println("\n*** RELIEF selected features ***\nFeature\tScore\n" + outR)
  }
  
  def evaluateClsPerformance(df: DataFrame, fsmodel: InfoThSelectorModel) = {
    val reducedData = fsmodel.transform(df)
    println("schema: " + reducedData.schema)
    // Train a NaiveBayes model.
    val model = new NaiveBayes()
      .setFeaturesCol("selectedFeatures")
      .setLabelCol(clsLabel)
      
    // Select example rows to display.
    val predictions = model.fit(reducedData).transform(reducedData)
    predictions.show()
    
    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(clsLabel)
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    evaluator.evaluate(predictions)
  }
  
  def kCVPerformance(df: DataFrame, fsmodel: InfoThSelectorModel, classifier: String) = {
    
    var inputCol = "selectedFeatures"
    var labelCol = clsLabel
    val reducedData = if(fsmodel != null) {
      fsmodel.transform(df)
    } else {
      inputCol = inputLabel
      df
    }
    reducedData.show()
    println("Reduced schema: " + reducedData.schema)
    val splits = MLUtils.kFold(reducedData.rdd, 10, seed)
    val sql = df.sqlContext
        
    val estimator = if(classifier == "nb") {
       new NaiveBayes()
        .setFeaturesCol(inputCol)
        .setLabelCol(clsLabel)    
     
    } else if(classifier == "dt") {
      val labelIndexer = new StringIndexer()
          .setInputCol(labelCol)
          .setOutputCol("indexedLabel")
          .fit(reducedData)
      labelCol = "indexedLabel"    
      // Automatically identify categorical features, and index them.
      val featureIndexer = new VectorIndexer()
        .setInputCol(inputCol)
        .setOutputCol("indexedFeatures")
        .setMaxCategories(15) // features with > 4 distinct values are treated as continuous.
        .fit(reducedData)
        
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
    val sum = splits.zipWithIndex.map { case ((training, validation), _) =>
      val trainingDataset = sql.createDataFrame(training, reducedData.schema).cache()
      val validationDataset = sql.createDataFrame(validation, reducedData.schema).cache()

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
    while (selected.size < nselect && moreFeat) {

      attAlive(selected.head.feat) = false
      val redundancies = redundancyMatrix(::, selected.head.feat)
              .toArray
              .dropRight(1)
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
        .slice(0, nselect).map{ case(id, score) => F(id,score)}.toSeq
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
    val featureAssembler = new VectorAssembler()
      .setInputCols(newNames)
      .setOutputCol(inputLabel)

    var processedDF = featureAssembler.transform(cleanedDF)
      .select(clsLabel, inputLabel)

    println("clsLabel: " + clsLabel)
    println("Columns: " + processedDF.columns.mkString(","))
    println("Schema: " + processedDF.schema.toString)
    println(processedDF.first.get(1))
      
    if(discretize){      
      val discretizer = new MDLPDiscretizer()
        .setMaxBins(15)
        .setMaxByPart(10000)
        .setInputCol(inputLabel)
        .setLabelCol(clsLabel)
        .setOutputCol("disc-" + inputLabel)
        
      inputLabel = "disc-" + inputLabel
      
      val model = discretizer.fit(processedDF)
      processedDF = model.transform(processedDF)
      processedDF.show()
      continuous = false
    } else if(continuous) {
      val scaler = new MinMaxScaler()
        .setInputCol(inputLabel)
        .setOutputCol("norm-" + inputLabel)

      inputLabel = "norm-" + inputLabel
      processedDF = scaler.fit(processedDF).transform(processedDF)
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
  
  def fitMRMR(df: DataFrame) = {
    val selector = new InfoThSelector()
        .setSelectCriterion("mrmr")
        .setNPartitions(nPartitions)
        .setNumTopFeatures(nselect)
        .setFeaturesCol(inputLabel) // this must be a feature vector
        .setLabelCol(clsLabel)
        .setOutputCol("selectedFeatures")
        
    selector.fit(df)
  }
  
  def log2(x: Double) = { math.log(x) / math.log(2) }
}
