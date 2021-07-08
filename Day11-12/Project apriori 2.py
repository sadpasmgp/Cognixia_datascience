# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir('Give Path')

# Data Preprocessing
dataset = pd.read_csv('groceries.csv', header = None)
transactions = []
for i in range(0, 9835):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 32)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.001, min_confidence = 0.3, min_lift = 5, min_length = 2)

# Visualising the results
results = list(rules)

