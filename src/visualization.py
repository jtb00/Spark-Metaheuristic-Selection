import re
import pandas as pd
import matplotlib.pyplot as plt

path = 'cols2.txt'
file = open(path, 'r')
cols = file.read()
cols = re.sub(r'(\[|\]|\')', '', cols)
cols = re.sub('\n', ', ', cols)
cols = cols.split(',')
# print(cols)

counts = pd.Series(cols).value_counts()
counts = counts.drop(labels=' ')
# print(counts)
counts.plot(kind='bar', title='Features Dropped by Simulated Annealing Algorithm')
plt.ylabel('# Times Dropped')
plt.show()
