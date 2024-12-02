import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, mean, median, mode, max, isnull
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer


# Removes any features with more than 1% missing values, returns modified dataframe and list of features with <1%
# missing values
def null_features(df):
    features = df.columns
    percents = []
    num_rows = df.count()
    for feature in features:
        count = df.where(col(feature).isNull()).count()
        percent = float(count) / num_rows
        percents.append(percent)
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
            si = StringIndexer(inputCol=feature, outputCol=feature + '_INDEXED', handleInvalid='keep')
            indexers.append(si)
            to_drop.append(feature)
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)
    df = df.drop(*to_drop)
    for feature in to_drop:
        df = df.withColumn(feature + '_INDEXED', df[feature + '_INDEXED'].cast('int'))
    return df


def fill_nulls(df, int_mode, double_mode):
    fill = {}
    features = df.columns
    for feature in features:
        if df.filter(isnull(col(feature))).count() > 0:
            # Workaround for indexed string columns; because StringIndexer has no option to preserve null values, these
            # were set equal to the number of different labels. This replaces all rows with this value with null.
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

# Drop unneeded columns
POS_CASH_balance = POS_CASH_balance.drop('SK_ID_CURR')
credit_card_balance = credit_card_balance.drop('SK_ID_CURR')
installments_payments = installments_payments.drop('SK_ID_CURR')

# Remove any features with > 1% null values
application = null_features(application)
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

# Sampling

# Feature engineering

# Get most recent previous credit for each loan from other institution(s)
w = Window.partitionBy('SK_ID_CURR')
bureau = bureau.withColumn('DAYS_CREDIT_MOST_RECENT', max('DAYS_CREDIT').over(w))
bureau = bureau.where(col('DAYS_CREDIT') == col('DAYS_CREDIT_MOST_RECENT')).drop('DAYS_CREDIT_MOST_RECENT')

# Get most recent monthly balance for each credit from other institution(s)
w = Window.partitionBy('SK_ID_BUREAU')
bureau_balance = bureau_balance.withColumn('MONTHS_BALANCE_MOST_RECENT', max('MONTHS_BALANCE').over(w))
bureau_balance = bureau_balance.where(col('MONTHS_BALANCE') == col('MONTHS_BALANCE_MOST_RECENT'))
bureau_balance = bureau_balance.drop('MONTHS_BALANCE_MOST_RECENT')

# Get most recent previous credit for each loan from Home Credit
w = Window.partitionBy('SK_ID_CURR')
previous_application = previous_application.withColumn('DAYS_DECISION_MOST_RECENT', max('DAYS_DECISION').over(w))
previous_application = previous_application.where(col('DAYS_DECISION') == col('DAYS_DECISION_MOST_RECENT')).drop('DAYS_DECISION_MOST_RECENT')

# Get most recent POS/cash balance for each previous credit from Home Credit
w = Window.partitionBy('SK_ID_PREV')
POS_CASH_balance = POS_CASH_balance.withColumn('MONTHS_BALANCE_MOST_RECENT', max('MONTHS_BALANCE').over(w))
POS_CASH_balance = POS_CASH_balance.where(col('MONTHS_BALANCE') == col('MONTHS_BALANCE_MOST_RECENT')).drop('MONTHS_BALANCE_MOST_RECENT')

# Get most recent credit card balance for each previous credit from Home Credit
w = Window.partitionBy('SK_ID_PREV')
credit_card_balance = credit_card_balance.withColumn('MONTHS_BALANCE_MOST_RECENT', max('MONTHS_BALANCE').over(w))
credit_card_balance = credit_card_balance.where(col('MONTHS_BALANCE') == col('MONTHS_BALANCE_MOST_RECENT')).drop('MONTHS_BALANCE_MOST_RECENT')

# Join dataframes
bureau = bureau.join(bureau_balance, "SK_ID_BUREAU", 'left')
application = application.join(bureau, "SK_ID_CURR", 'left')
application = application.join(previous_application, "SK_ID_CURR", 'left')

# Handle remaining missing values
application = fill_nulls(application, 'mode', 'mean')
credit_card_balance = fill_nulls(credit_card_balance, 'mode', 'mean')
installments_payments = fill_nulls(installments_payments, 'mode', 'mean')
POS_CASH_balance = fill_nulls(POS_CASH_balance, 'mode', 'mean')
previous_application = fill_nulls(previous_application, 'mode', 'mean')

# Feature selection
