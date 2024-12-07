from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, array, lit
from pyspark.ml.feature import VectorAssembler


# Sampling

input_path = '/user/jbedinge/output/fe/*'
output_path = '/user/jbedinge/output/sampled'
master = 'yarn'

spark = (SparkSession.builder.master(master).appName('Home Credit').config('spark.jars', 'approx-smote-1.1.2.jar')
         .getOrCreate())
sc = SparkContext.getOrCreate()
sc.setLogLevel('ERROR')

application = spark.read.csv(input_path, header=True, inferSchema=True)

seed = 0
sample_method = 'oversample'

num_true = application.where(col('TARGET') == 1).count()
num_false = application.where(col('TARGET') == 0).count()
if sample_method == 'undersample':
    print('Undersampling...')
    ratio = num_true / num_false
    application = application.sampleBy('TARGET', {0: ratio, 1: 1}, seed=seed)
elif sample_method == 'oversample':
    print('Oversampling...')
    ratio = num_false / num_true
    print(application.count())
    application_true = application.filter(col('TARGET') == 1)
    application_false = application.filter(col('TARGET') == 0)
    application_true = application_true.withColumn('temp', explode(array([lit(x) for x in range(int(ratio))])))
    application_true = application_true.drop('temp')
    application = application_true.union(application_false)
    print(application.count())
else:
    print('Performing SMOTE...')

application.write.option('header', True).mode('overwrite').csv(output_path)
