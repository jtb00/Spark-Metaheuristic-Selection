import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, mean, median, mode, max, isnull, when, sum
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from functools import reduce
from operator import add


# Removes any features with more than 1% missing values, returns modified dataframe
def null_features(df):
    features = df.columns
    percents = []
    num_rows = df.count()
    for feature in features:
        count = df.where(col(feature).isNull()).count()
        percent = float(count) / num_rows
        if percent >= 0.01:
            df = df.drop(feature)
    return df


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
    return df


def fill_nulls(df):
    # print(df.dtypes)
    fill = {}
    features = df.columns
    features.remove('TARGET')
    features.remove('SK_ID_CURR')
    # print(features)
    indexeds = [feature for feature in features if '_INDEXED' in feature]
    # has_nulls = [feature for feature in features if df.filter(isnull(col(feature))).count() > 0]
    modes = df.select([mode(x).alias(x) for x in features])
    means = df.select([mean(x).alias(x) for x in features])
    max_vals = df.select([max(x).alias(x) for x in features])
    for feature in features:
        if feature in indexeds:
            fill[feature] = max_vals.collect()[0][feature]
        elif feature in application_cols:
            if df.dtypes[df.columns.index(feature)][1] == 'int':
                fill[feature] = modes.collect()[0][feature]
            else:
                fill[feature] = means.collect()[0][feature]
        else:
            fill[feature] = 0
    print(fill)
    df = df.na.fill(fill)
    return df 


master = 'local[*]'
input_path = 'home-credit-default-risk'
output_path = '/user/jbedinge/output'

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

# Drop unneeded columns

# These seem to be redundant
application = application.drop('FLAG_WORK_PHONE', 'REGION_RATING_CLIENT')

# Not needed, will merge with application later
POS_CASH_balance = POS_CASH_balance.drop('SK_ID_CURR')
credit_card_balance = credit_card_balance.drop('SK_ID_CURR')
installments_payments = installments_payments.drop('SK_ID_CURR')

# Remove any features with > 1% null values
print('Removing features with > 1% null values...')
application = null_features(application)
# print(application.columns)
bureau = null_features(bureau)
# print(bureau.columns)
bureau_balance = null_features(bureau_balance)
credit_card_balance = null_features(credit_card_balance)
# print(credit_card_balance.columns)
POS_CASH_balance = null_features(POS_CASH_balance)
# print(POS_CASH_balance.columns)
previous_application = null_features(previous_application)
# print(previous_application.columns)

application = index_string_cols(application)
# print(application.dtypes)
bureau = index_string_cols(bureau)
bureau_balance = index_string_cols(bureau_balance)
credit_card_balance = index_string_cols(credit_card_balance)
POS_CASH_balance = index_string_cols(POS_CASH_balance)
previous_application = index_string_cols(previous_application)

# Feature engineering
print('Feature engineering...')

# Combine FLAG_DOCUMENT columns into single column
doc_cols = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
            'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
            'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
            'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
application = application.withColumn('FLAG_DOCUMENTS', reduce(add, [col(x) for x in doc_cols]))
application = application.drop(*doc_cols).cache()
# Get list of features originating from this CSV file
application_cols = application.columns

# Get most recent monthly balance for each credit from other institution(s)
w = Window.partitionBy('SK_ID_BUREAU')
bureau_balance = bureau_balance.withColumn('MONTHS_BALANCE_MOST_RECENT', max('MONTHS_BALANCE').over(w))
bureau_balance = bureau_balance.where(col('MONTHS_BALANCE') == col('MONTHS_BALANCE_MOST_RECENT'))
bureau_balance = bureau_balance.drop('MONTHS_BALANCE_MOST_RECENT').cache()

# Get most recent previous credit for each loan from other institution(s)
w = Window.partitionBy('SK_ID_CURR')
bureau = bureau.join(bureau_balance, "SK_ID_BUREAU", 'left').drop('SK_ID_BUREAU')
bureau = bureau.withColumn('DAYS_CREDIT_MOST_RECENT', max('DAYS_CREDIT').over(w))
bureau = bureau.where(col('DAYS_CREDIT') == col('DAYS_CREDIT_MOST_RECENT')).drop('DAYS_CREDIT_MOST_RECENT')
bureau = bureau.withColumn('DAYS_CREDIT_UPDATE_MOST_RECENT', max('DAYS_CREDIT_UPDATE').over(w))
bureau = bureau.where(col('DAYS_CREDIT_UPDATE') == col('DAYS_CREDIT_UPDATE_MOST_RECENT'))
bureau = bureau.drop('DAYS_CREDIT_UPDATE_MOST_RECENT')
# Average any remaining duplicate rows
dups = bureau.groupBy("SK_ID_CURR").count().where("count > 1").drop("count")
dups = bureau.join(dups, 'SK_ID_CURR', 'leftsemi')
dups = dups.groupBy("SK_ID_CURR").agg(*[mean(x).alias(x) for x in dups.columns if x != 'SK_ID_CURR'])
bureau = bureau.join(dups, 'SK_ID_CURR', 'leftanti')
bureau = bureau.union(dups)

# Get most recent previous credit for each loan from Home Credit
w = Window.partitionBy('SK_ID_CURR')
previous_application = previous_application.withColumn('DAYS_DECISION_MOST_RECENT', max('DAYS_DECISION').over(w))
previous_application = previous_application.where(col('DAYS_DECISION') == col('DAYS_DECISION_MOST_RECENT'))
previous_application = previous_application.drop('DAYS_DECISION_MOST_RECENT')

# Get most recent POS/cash balance for each previous credit from Home Credit
w = Window.partitionBy('SK_ID_PREV')
POS_CASH_balance = POS_CASH_balance.withColumn('MONTHS_BALANCE_MOST_RECENT', max('MONTHS_BALANCE').over(w))
POS_CASH_balance = POS_CASH_balance.where(col('MONTHS_BALANCE') == col('MONTHS_BALANCE_MOST_RECENT'))
POS_CASH_balance = POS_CASH_balance.drop('MONTHS_BALANCE_MOST_RECENT')

# Get most recent credit card balance for each previous credit from Home Credit
w = Window.partitionBy('SK_ID_PREV')
credit_card_balance = credit_card_balance.withColumn('MONTHS_BALANCE_MOST_RECENT', max('MONTHS_BALANCE').over(w))
credit_card_balance = credit_card_balance.where(col('MONTHS_BALANCE') == col('MONTHS_BALANCE_MOST_RECENT'))
credit_card_balance = credit_card_balance.drop('MONTHS_BALANCE_MOST_RECENT')

# Get number of payments / missed payments
cond = (col('DAYS_ENTRY_PAYMENT') > col('DAYS_INSTALMENT')) | (col('AMT_INSTALMENT') > col('AMT_PAYMENT'))
installments_payments = installments_payments.withColumn('MISSED_PAYMENT', when(cond, 1).otherwise(0))
installments_payments = installments_payments.groupBy('SK_ID_PREV').agg(sum('MISSED_PAYMENT')
                                                                        .alias('NUM_MISSED_PAYMENTS'))
installments_payments = installments_payments.withColumn('NUM_MISSED_PAYMENTS', col('NUM_MISSED_PAYMENTS').cast('int'))

# Join dataframes
print('Joining dataframes...')
# print(application.count())
application = application.join(bureau, "SK_ID_CURR", 'left')
# print(application.count())

# Rename any potentially amiguous features
POS_CASH_balance = POS_CASH_balance.withColumnRenamed('MONTHS_BALANCE', 'MONTHS_BALANCE_POS_CASH')
credit_card_balance = credit_card_balance.withColumnRenamed('MONTHS_BALANCE', 'MONTHS_BALANCE_CREDIT_CARD')
POS_CASH_balance = POS_CASH_balance.withColumnRenamed('NAME_CONTRACT_STATUS_INDEXED', 'NAME_CONTRACT_STATUS_POS_CASH_INDEXED')
credit_card_balance = credit_card_balance.withColumnRenamed('NAME_CONTRACT_STATUS_INDEXED', 'NAME_CONTRACT_STATUS_CREDIT_CARD_INDEXED')
previous_application = previous_application.withColumnRenamed('NAME_CONTRACT_STATUS_INDEXED', 'NAME_CONTRACT_STATUS_PREV_APP_INDEXED')
POS_CASH_balance = POS_CASH_balance.withColumnRenamed('SK_DPD', 'SK_DPD_POS_CASH')
credit_card_balance = credit_card_balance.withColumnRenamed('SK_DPD', 'SK_DPD_CREDIT_CARD')
POS_CASH_balance = POS_CASH_balance.withColumnRenamed('SK_DPD_DEF', 'SK_DPD_DEF_POS_CASH').cache()
credit_card_balance = credit_card_balance.withColumnRenamed('SK_DPD_DEF', 'SK_DPD_DEF_CREDIT_CARD').cache()

previous_application = previous_application.join(POS_CASH_balance, "SK_ID_PREV", 'left')
previous_application = previous_application.join(credit_card_balance, "SK_ID_PREV", 'left')
previous_application = previous_application.join(installments_payments, "SK_ID_PREV", 'left')
previous_application = previous_application.drop('SK_ID_PREV')
dups = previous_application.groupBy("SK_ID_CURR").count().where("count > 1").drop("count")
dups = previous_application.join(dups, 'SK_ID_CURR', 'leftsemi')
dups_ints = dups.groupBy("SK_ID_CURR").agg(*[mode(x).alias(x) for x in dups.columns if x != 'SK_ID_CURR' and dups.dtypes[dups.columns.index(x)][1] == 'int'])
dups_doubles = dups.groupBy("SK_ID_CURR").agg(*[mean(x).alias(x) for x in dups.columns if x != 'SK_ID_CURR' and dups.dtypes[dups.columns.index(x)][1] == 'double'])
dups = dups_ints.join(dups_doubles, 'SK_ID_CURR', 'left')
previous_application = previous_application.join(dups, 'SK_ID_CURR', 'leftanti')
previous_application = previous_application.union(dups)
for feature in previous_application.columns:
    if feature in application.columns and feature != 'SK_ID_CURR':
        previous_application = previous_application.withColumnRenamed(feature, feature + '_PREV')
previous_application = previous_application.drop('SK_ID_PREV').cache()
application = application.join(previous_application, "SK_ID_CURR", 'left')
# print(application.count())

# Handle remaining missing values
print('Handling missing values...')
application = fill_nulls(application)

# application.write.option('header', True).mode('overwrite').csv(output_path + '/fe')
