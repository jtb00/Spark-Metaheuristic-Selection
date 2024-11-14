import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer


# Removes any features with more than 1% missing values, returns modified dataframe and list of features with <1%
# missing values
def null_features(df):
    features = df.columns
    percents = []
    num_rows = df.count()
    # print('Feature\tNull Values\tPercent Null')
    for feature in features:
        count = df.where(col(feature).isNull()).count()
        percent = float(count) / num_rows
        percents.append(percent)
        # print(feature + '\t' + str(count) + '\t' + str(round(percent, 2)))
    for i in range(len(features)):
        if percents[i] >= 0.01:
            df = df.drop(features[i])
    return df


# Use StringIndexer to map any categorical string columns to a set of integer values
def index_string_cols(df):
    features = df.columns
    to_drop = []
    indexers = []
    for feature in features:
        if df.dtypes[df.columns.index(feature)][1] == 'string':
            si = StringIndexer(inputCol=feature, outputCol=feature + '_INDEXED')
            indexers.append(si)
            to_drop.append(feature)
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)
    df = df.drop(*to_drop)
    for feature in to_drop:
        df = df.withColumn(feature + '_INDEXED', df[feature + '_INDEXED'].cast('int'))
    return df


master = 'local[*]'
input_path = 'home-credit-default-risk'

spark = SparkSession.builder.master(master).appName('Home Credit').getOrCreate()
sc = SparkContext.getOrCreate()
sc.setLogLevel('ERROR')

print('Importing data...')
application = spark.read.csv(input_path + '/application_train.csv', header=True, inferSchema=True)
bureau = spark.read.csv(input_path + '/bureau.csv', header=True, inferSchema=True)
bureau_balance = spark.read.csv(input_path + '/bureau_balance.csv', header=True, inferSchema=True)
credit_card_balance = spark.read.csv(input_path + '/credit_card_balance.csv', header=True, inferSchema=True)
installments_payments = spark.read.csv(input_path + '/installments_payments.csv', header=True, inferSchema=True)
POS_CASH_balance = spark.read.csv(input_path + '/POS_CASH_balance.csv', header=True, inferSchema=True)
previous_application = spark.read.csv(input_path + '/previous_application.csv', header=True, inferSchema=True)
# print(installments_payments.count())

# Remove any features with > 1% null values
application = null_features(application)
# print(application_nulls)
bureau = null_features(bureau)
bureau_balance = null_features(bureau_balance)
credit_card_balance = null_features(credit_card_balance)
installments_payments = null_features(installments_payments)
POS_CASH_balance = null_features(POS_CASH_balance)
previous_application = null_features(previous_application)

application = index_string_cols(application)
bureau = index_string_cols(bureau)
bureau_balance = index_string_cols(bureau_balance)
credit_card_balance = index_string_cols(credit_card_balance)
previous_application = index_string_cols(previous_application)
previous_application.printSchema()

# Handle remaining missing values

# Feature engineering

# Join dataframes

# Sampling

# Feature selection
