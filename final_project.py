import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, median, mode, max
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
    null_features = []
    for i in range(len(features)):
        if percents[i] >= 0.01:
            df = df.drop(features[i])
        elif percents[i] != 0.0:
            # Allows for performing operations on specifically these columns later
            if df.dtypes[df.columns.index(features[i])][1] == 'string':
                null_features.append(features[i] + '_INDEXED')
            else:
                null_features.append(features[i])
    return df, null_features


# Use StringIndexer to map any categorical string columns to a set of integer values
def index_string_cols(df):
    features = df.columns
    to_drop = []
    indexers = []
    for feature in features:
        if df.dtypes[df.columns.index(feature)][1] == 'string':
            si = StringIndexer(inputCol=feature, outputCol=feature + '_INDEXED', handleInvalid='keep')
            indexers.append(si)
            to_drop.append(feature)
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)
    df = df.drop(*to_drop)
    for feature in to_drop:
        df = df.withColumn(feature + '_INDEXED', df[feature + '_INDEXED'].cast('int'))
        # df = df.withColumnRenamed(feature + '_INDEXED', feature)
    return df


def fill_nulls(df, nulls, int_mode, double_mode):
    fill = {}
    for feature in nulls:
        # Workaround for indexed string columns; because StringIndexer has no option to preserve null values, these were
        # set equal to the number of different labels. This replaces all rows with this value with null.
        if '_INDEXED' in feature:
            max_val = df.select(max(feature)).collect()[0][0]
            df = df.na.replace(max_val, None, feature)
        if df.dtypes[df.columns.index(feature)][1] == 'integer':
            if int_mode == 'median':
                val = df.agg(median(feature)).collect()[0][0]
            elif int_mode == 'mode':
                val = df.agg(mode(feature)).collect()[0][0]
            else:
                val = df.agg(mean(feature)).collect()[0][0]
        else:
            if double_mode == 'median':
                val = df.agg(median(feature)).collect()[0][0]
            elif double_mode == 'mode':
                val = df.agg(mode(feature)).collect()[0][0]
            else:
                val = df.agg(mean(feature)).collect()[0][0]
        fill[feature] = val
    df = df.na.fill(fill)
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
# print(previous_application.first())

# Remove any features with > 1% null values
application, application_nulls = null_features(application)
# print(application_nulls)
bureau, bureau_nulls = null_features(bureau)
bureau_balance, bureau_balance_nulls = null_features(bureau_balance)
credit_card_balance, credit_card_balance_nulls = null_features(credit_card_balance)
installments_payments, installments_payments_nulls = null_features(installments_payments)
POS_CASH_balance, POS_CASH_balance_nulls = null_features(POS_CASH_balance)
previous_application, previous_application_nulls = null_features(previous_application)
# print(previous_application.first())

application = index_string_cols(application)
bureau = index_string_cols(bureau)
bureau_balance = index_string_cols(bureau_balance)
credit_card_balance = index_string_cols(credit_card_balance)
previous_application = index_string_cols(previous_application)
# print(previous_application.first())

# Handle remaining missing values
application = fill_nulls(application, application_nulls, 'mode', 'mean')
print(application.filter(application[application_nulls[0]].isNull()).count())

# Feature engineering

# Join dataframes

# Sampling

# Feature selection
