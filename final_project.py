from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# Removes any features with more than 1% missing values, returns modified dataframe and list of features with <1%
# missing values
def null_features(df):
    features = df.columns
    percents = []
    num_rows = df.count()
    print('Feature\tNull Values\tPercent Null')
    for feature in features:
        count = df.where(col(feature).isNull()).count()
        percent = float(count) / num_rows
        percents.append(percent)
        print(feature + '\t' + str(count) + '\t' + str(round(percent, 2)))
    null_features = []
    for i in range(len(features)):
        if percents[i] >= 0.01:
            df = df.drop(features[i])
        elif percents[i] != 0.0:
            null_features.append(features[i])
    print('Features removed: ' + str(len(features) - len(df.columns)))
    return df, null_features


master = 'local[*]'
input_path = 'home-credit-default-risk'

spark = SparkSession.builder.master(master).appName('Home Credit').getOrCreate()
sc = SparkContext.getOrCreate()
sc.setLogLevel('ERROR')

print('Importing data...')
application = spark.read.option('header', True).csv(input_path + '/application_train.csv')
bureau = spark.read.option('header', True).csv(input_path + '/bureau.csv')
bureau_balance = spark.read.option('header', True).csv(input_path + '/bureau_balance.csv')
credit_card_balance = spark.read.option('header', True).csv(input_path + '/credit_card_balance.csv')
installments_payments = spark.read.option('header', True).csv(input_path + '/installments_payments.csv')
POS_CASH_balance = spark.read.option('header', True).csv(input_path + '/POS_CASH_balance.csv')
previous_application = spark.read.option('header', True).csv(input_path + '/previous_application.csv')
# print(installments_payments.count())

# Remove any features with > 1% null values
application, application_nulls = null_features(application)
# print(application_nulls)
bureau, bureau_nulls = null_features(bureau)
bureau_balance, bureau_balance_nulls = null_features(bureau_balance)
credit_card_balance, credit_card_balance_nulls = null_features(credit_card_balance)
installments_payments, installments_payments_nulls = null_features(installments_payments)
POS_CASH_balance, POS_CASH_balance_nulls = null_features(POS_CASH_balance)
previous_application, previous_application_nulls = null_features(previous_application)
