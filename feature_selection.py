import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession, Window


def metrics(predictions):
    tp = predictions.filter((predictions.label == 1) & (predictions.prediction == 1)).count()
    tn = predictions.filter((predictions.label == 0) & (predictions.prediction == 0)).count()
    fp = predictions.filter((predictions.label == 0) & (predictions.prediction == 1)).count()
    fn = predictions.filter((predictions.label == 1) & (predictions.prediction == 0)).count()
    accuracy = (tp + tn) / float(predictions.count())
    if (tp + fp) > 0:
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
    else:
        precision = 0
        recall = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return accuracy, precision, recall, f1


master = 'local[*]'
input_path = 'home-credit-default-risk'

spark = SparkSession.builder.master(master).appName('Home Credit').getOrCreate()
sc = SparkContext.getOrCreate()
sc.setLogLevel('ERROR')

# Feature selection

seed = 0
selection = 'hill_climb'
