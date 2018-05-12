# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) # header argument specifies no titles in dataset

# prepare input for apriori function, must build list of lists
transactions = [] # intitialize list variable, empty vector
for i in range(0, 7501): # loop through all the transactions
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) # takes each item in transaction, contains it in a list as str type

# apyori script serves as the package
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2) # keyword arguments depend on problem and data
# support started at 3 * 7 / 7500 whics is 3 times a day times 7 days a week divided by total transactions in a week.
# High confidence can lead to obvious results, so play with the number, it can be arbitrary

# Visualising the results
results = list(rules)