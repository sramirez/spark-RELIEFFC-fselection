RELIEF-F feature selection for Apache Spark
=====================================================

The present algorithm (called BELIEF) implements Feature Weighting (FW) on Spark for its application on Big Data problems. This repository contains an improved implementation of RELIEF-F algorithm [1], which has been extended with a cheap but effective feature redundancy elimination technique. BELIEF leverages distance computations computed in prior steps to estimate inter-feature redundancy relationships at virtually no cost. BELIEF is also highly scalable to different sample sizes, from hundreds of samples to thousands. 

Spark package: https://spark-packages.org/package/sramirez/spark-RELIEFFC-fselection.

## Main features:

* Compliance with 2.2.0 Spark version, and ml API.
* Support for sparse data and high-dimensional datasets (millions of features).
* Include a new heuristic that removes redundant features from the final selection set.
* Scalable to large sample sets.

This software has been tested on several large-scale datasets, such as:

- Oversampled ECBDL14 dataset (64M instances, 631 features): a dataset selected for the GECCO-2014 in Vancouver, July 13th, 2014 competition, which comes from the Protein Structure Prediction field (http://cruncher.ncl.ac.uk/bdcomp/). 
- kddb dataset (20M instances, nearly 30M of features): http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2010%20%28bridge%20to%20algebra%29.

## Example (ml): 
	import org.apache.spark.ml.feature._
	val selector = new ReliefFRSelector()
        	.setNumTopFeatures(10)
		.setEstimationRatio(0.1) 
        	.setSeed(123456789L) // for sampling
		.setNumNeighbors(5) // k-NN used in RELIEF
        	.setDiscreteData(true)
        	.setInputCol("features")// this must be a feature vector
        	.setLabelCol("labelCol")
        	.setOutputCol("selectedFeatures")


	val result = selector.fit(df).transform(df)

## Prerequisites:

Continuous data must have 0 mean, and 1 std to achieve a better performing in REDUNDANCY estimations.
Standard scaler in ML library may be used to fulfill this recommendation:

https://spark.apache.org/docs/latest/ml-features.html#standardscaler

## Contributors

- Sergio Ramírez-Gallego (sramirez@decsai.ugr.es) (main contributor and maintainer).

## References

[1] I. Kononenko, E. Simec, M. Robnik-Sikonja, Overcoming the myopia of inductive learning algorithms with RELIEFF, Applied Intelligence 7 (1) (1997) 39–55.
