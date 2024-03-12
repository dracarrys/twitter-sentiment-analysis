#import all the libraries of pyspark.sql
from pyspark.sql import*
#import SparkContext and SparkConf
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F
import os
os.environ["JAVA_HOME"] = "C:/Java/jdk1.8.0_281"
os.environ["SPARK_HOME"] = "C:/Spark/spark-3.0.2-bin-hadoop2.7"
import findspark
findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()
#setup configuration property
#set the master URL
#set an application name
conf = SparkConf()
#start spark cluster
#if already started then get it else start it
sc = SparkContext.getOrCreate(conf=conf)
#initialize SQLContext from spark cluster
sqlContext = SQLContext(sc)

#Filepath variable for your file location directory
FilePath="/home/kosicsd/PycharmProjects/sentiment/trainingandtestdata/"
#FileName variable
FileNameTrain="train.csv"
FileNameTest="test.csv"


#combine both above variables
FullPathTrain= FilePath + FileNameTrain
FullPathTest= FilePath + FileNameTest

#dataframe
#set header property true for the actual header columns
train_set=sqlContext.read.csv(FullPathTrain, header=False)
test_set=sqlContext.read.csv(FullPathTest, header=False)
#display data from the dataframe
train_set.show()
test_set.show()


train_set = train_set.withColumnRenamed('_c0', 'Polarity')
test_set = test_set.withColumnRenamed('_c0', 'Polarity')

train_set = train_set.withColumnRenamed('_c1', 'ID')
test_set = test_set.withColumnRenamed('_c1', 'ID')

train_set = train_set.withColumnRenamed('_c5', 'Text')
test_set = test_set.withColumnRenamed('_c5', 'Text')

columns_to_drop = ['_c2','_c3', '_c4']

train_set = train_set.drop(*columns_to_drop)
test_set = test_set.drop(*columns_to_drop)

train_set = train_set.dropna()
test_set = test_set.dropna()

print(train_set.count(), test_set.count())
def preprocessing(words):
    words = words.na.replace('', None)
    words = words.na.drop()
    words = words.withColumn('Text', F.regexp_replace('Text', r'http\S+', ''))
    words = words.withColumn('Text', F.regexp_replace('Text', '@\w+', ''))
    words = words.withColumn('Text', F.regexp_replace('Text', '#', ''))
    words = words.withColumn('Text', F.regexp_replace('Text', 'RT', ''))
    words = words.withColumn('Text', F.regexp_replace('Text', ':', ''))
    return words

train_set = preprocessing(train_set)
test_set = preprocessing(test_set)
train_set.show()
test_set.show()

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="Text", outputCol="words")
hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "Polarity", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

pipelineFit = pipeline.fit(train_set)
train_df = pipelineFit.transform(train_set)
test_df = pipelineFit.transform(test_set)

train_df.show(5)

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(maxDepth=5)
rfModel = rf.fit(train_df)
predictions = rfModel.transform(test_df)
#print(predictions)


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='accuracy')

accuracy = evaluator.evaluate(predictions)
print('Test Accuracy = ', accuracy)