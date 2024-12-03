from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, array, lit
from pyspark.ml.feature import VectorAssembler

# Sampling

input_path = ''
master = 'local[*]'

spark = SparkSession.builder.master(master).appName('Home Credit').getOrCreate()
sc = SparkContext.getOrCreate()
sc.setLogLevel('ERROR')

application = spark.read.csv(input_path + '/application/*', header=True, inferSchema=True)

seed = 0
sample_method = 'undersample'

print('Sampling...')
num_true = application.where(col('TARGET') == 1).count()
num_false = application.where(col('TARGET') == 0).count()
ratio = num_true / num_false
if sample_method == 'undersample':
    application = application.sampleBy('TARGET', {0: ratio, 1: 1}, seed=seed)
elif sample_method == 'oversample':
    application_true = application.filter(col('TARGET') == 1)
    application_false = application.filter(col('TARGET') == 0)
    application_true = application_true.withColumn('temp', explode(array([lit(x) for x in range(int(ratio))])))
    application_true = application_true.drop('temp')
