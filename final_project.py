from pyspark import SparkContext
from pyspark.sql import SparkSession

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
print(installments_payments.count())
