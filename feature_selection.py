import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession, Window
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, LinearSVC
from pyspark.ml.feature import VectorAssembler


def metrics(predictions, metric):
    tp = predictions.filter((predictions.label == 1) & (predictions.prediction == 1)).count()
    tn = predictions.filter((predictions.label == 0) & (predictions.prediction == 0)).count()
    fp = predictions.filter((predictions.label == 0) & (predictions.prediction == 1)).count()
    fn = predictions.filter((predictions.label == 1) & (predictions.prediction == 0)).count()
    accuracy = (tp + tn) / float(predictions.count())
    if metric == 'accuracy':
        return accuracy
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
    return f1


def generate_neighbors(num_neighbors, cols, num_change):
    neighbors = []
    for i in range(num_neighbors):
        neighbor = cols.copy()
        flip_cols = np.random.choice(list(cols.keys()), num_change, replace=False)
        for feature in flip_cols:
            neighbor[feature] = 1 if neighbor[feature] == 0 else 0
        neighbors.append(neighbor.copy())
    return neighbors


def hill_climb(classifier, metric):
    x = dict(zip(feature_cols, [1] * len(feature_cols)))
    i = 1
    va = VectorAssembler(inputCols=feature_cols, outputCol='features')
    df_train = va.transform(train)
    df_train = df_train.select('features', 'label')
    df_test = va.transform(test)
    df_test = df_test.select('features', 'label')
    predictions = classifier.fit(df_train).transform(df_test)
    best_score = metrics(predictions, metric)
    print(f'Best score: {best_score}')
    while True:
        print(f'Iteration {i}:')
        neighbors = generate_neighbors(5, x.copy(), 5)
        x_new = x
        for neighbor in neighbors:
            drop = list(k for k, v in neighbor.items() if v == 0)
            # print(drop)
            cols = list(k for k, v in neighbor.items() if v == 1)
            va = VectorAssembler(inputCols=cols, outputCol='features')
            df_train = va.transform(train)
            df_train = df_train.select('features', 'label')
            df_test = va.transform(test)
            df_test = df_test.select('features', 'label')
            predictions = classifier.fit(df_train).transform(df_test)
            score = metrics(predictions, metric)
            print(f'Score: {score}')
            if score > best_score:
                best_score = score
                x_new = neighbor
                break
        if x_new == x:
            print(f'Best score: {best_score}')
            break
        x = x_new
        i = i + 1 
    return x


master = 'yarn'
input_path = '/user/jbedinge/output/sampled/*'

spark = SparkSession.builder.master(master).appName('Home Credit').getOrCreate()
sc = SparkContext.getOrCreate()
sc.setLogLevel('ERROR')

application = spark.read.csv(input_path, header=True, inferSchema=True)
print(application.count())

# Feature selection

seed = 0
selection = 'hill_climb'
# classifier = 'logistic'
metric = 'accuracy'

classifiers = {
    'logistic': LogisticRegression(featuresCol='features', labelCol='label'),
    'decision_tree': DecisionTreeClassifier(featuresCol='features', labelCol='label', seed=seed),
    'random_forest': RandomForestClassifier(featuresCol='features', labelCol='label', seed=seed),
    'gradient_boost': GBTClassifier(featuresCol='features', labelCol='label', seed=seed),
    'linear_svc': LinearSVC(featuresCol='features', labelCol='label')
}

np.random.seed(seed)

application = application.withColumnRenamed('TARGET', 'label')
feature_cols = application.columns
feature_cols.remove('label')
feature_cols.remove('SK_ID_CURR')
print(feature_cols)
train, test = application.randomSplit([0.8, 0.2], seed=seed)
train = train.cache()
test = test.cache()

if selection == 'hill_climb':
    for name, classifier in classifiers.items():
        print(name + ':')
        cols = hill_climb(classifier, metric)
        dropped_cols = list(k for k, v in cols.items() if v == 0)
        print(dropped_cols)

