package comp9313.ass4
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.collection.mutable.HashMap

object SetSimJoin {
  def main (args: Array[String]) {
    
    // set the input file and output folder from the arguments
    val inputFile = args(0)
    val outputFolder = args(1)
    val tau = args(2).toDouble
    
    /// initialise configuration and spark context
    val conf = new SparkConf().setAppName("Set Similarity Joiner").setMaster("local")
    val sc = new SparkContext(conf)
    
    //read input file
    val input = sc.textFile(inputFile)
    
    // STAGE 1 
    // Basic Token Ordering (BDO)
    
    // Phase1: flat map all values, excluding the first item on each row (which is RID). Convert RIDs to Int. 
    val freqMap = input.flatMap(x => x.split(" ").drop(1)).map(x => (x.toInt,1))
    // Phase2: count the frequency of each token and then sort by frequency. This will enforce global sorting.  
    
    val orderList = freqMap.reduceByKey(_+_)
    val orderHash: HashMap[Int, Int] = HashMap.empty[Int, Int]
    orderList.map{p => orderHash.put(p._1, p._2) }
    // broadcast hashmap so each core can use the frequency list. 
    val broadcast = sc.broadcast(orderHash)
    
    // STAGE 2
    // Generate (k,v) pairs where k is RID and v is an array of tokens
    val rMap = input.map(x => (x.split(" ")(0).toInt, x.split(" ").drop(1).map(_.toInt)))
    // order tokens based of global ordering as in phase2. A hashmap is used for quick access times. 
    val orderMap = rMap.map(x => (x._1, x._2.sortBy(x => (broadcast.value.get(x), x) ) )) 
    
    // Prefix Filtering: emit (k,v) pairs where k is the token for the first p tokens. 
    val rrMap = orderMap.flatMap{x => 
      val prefixLength = x._2.length - math.ceil(x._2.length * tau).toInt + 1     
      for(i <- 0 until prefixLength ) yield {
        (x._2(i), x)
      }
    }
    
    // Group all pairs which share token, then join with other RIDs with same join attribute. 
    val rGroup = rrMap.groupByKey().flatMap( { case(t,u) =>
        val newpair = u.toArray
        for (i <- 0 until u.size-1;j <- i+1 until u.size) yield {
          val r = newpair(i)._2.toSet
            val s = newpair(j)._2.toSet
            // calculating similarity
            val sim = r.intersect(s).size.toDouble/r.union(s).size.toDouble
            ((newpair(i)._1, newpair(j)._1), sim)
          }
        }
    )
    
   // Stage 3
   // Reduce duplicate records, filter any records lower than the threshold, and then sort by RID pair.  
   val output = rGroup.reduceByKey((x,y) => Math.max(x,y)).filter(x => x._2 >= tau).sortByKey(true).map(x => x._1.toString() + "\t" + x._2.toString())
    
    //Save output
    output.saveAsTextFile(outputFolder)
      
    
  }
}