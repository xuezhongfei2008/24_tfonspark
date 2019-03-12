import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

spark=SparkSession\
.builder.\
appName("Word2Vec")\
.config("spark.some.config.option","some-value")\
.getOrCreate()

df = spark.read.text('hdfs://10.242.4.95:9000/user/gongxf/all_session_cut_0703.txt')

# 数据处理
f = udf(lambda x : x.split(' '), ArrayType(StringType()))
df = df.withColumn('sentences', f(df['value']))

word2vec = Word2Vec(vectorSize=300, minCount=50, inputCol="sentences", outputCol="result")
w2v_model = word2vec.fit(df)

print('done')
