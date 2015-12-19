import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by snehasis on 12/18/2015.
  */
object evalMetrics {
  def main(args: Array[String]) {


    /*spark stuff*/
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("INFO").setLevel(Level.OFF)
    System.setProperty("hadoop.home.dir", "c:/winutil/")
    val conf = new SparkConf().setAppName("MusicReco").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer").set("spark.executor.memory", "4g").setMaster("local[*]")
    val sc = new SparkContext(conf)

    /*setting up sql context to query the data later on*/
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    println("Spark Context started")
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("INFO").setLevel(Level.OFF)

    /*start logic*/
    val rawData = sc.textFile("dataset/covtype.data")
    val data = rawData.map { line =>
      val values = line.split(',').map(_.toDouble)
      val featureVector = Vectors.dense(values.init) //init returns all but last value; target is last column
    val label = values.last - 1 //DecisionTree needs labels starting at 0; subtract 1
      LabeledPoint(label, featureVector)
    }

    //splitting data
    val Array(trainData, cvData, testData) =
      data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()

    val evaluations =
      for (impurity <- Array("gini", "entropy");
           depth <- Array(1, 20);
           bins <- Array(10, 300))
        yield {
          val model = DecisionTree.trainClassifier(
            trainData, 7, Map[Int, Int](), impurity, depth, bins)
          val predictionsAndLabels = cvData.map(example =>
            (model.predict(example.features), example.label)
          )
          val accuracy =
            new MulticlassMetrics(predictionsAndLabels).precision
          ((impurity, depth, bins), accuracy)
        }

    evaluations.sortBy(_._2).reverse.foreach(println)

  }
}
