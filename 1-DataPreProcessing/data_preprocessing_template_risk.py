# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 20:03:49 2018

@author: info
"""
# Data Preprocessing Template

# Importing the libraries
# Three Fundamental Libraries
import numpy as np # Used for Mathematics
import matplotlib.pyplot as plt # Pyplot is a sub library for plotting graphs
import pandas as pd # Used to import and manage data.

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Create matrix of independent variables, aka matrix of features
# Country, Age, Salary columns are independent variables
X = dataset.iloc[:, :-1].values # All rows and cols, except last col

# Create dependent variable matrix for Purchased column
y = dataset.iloc[:, 3].values # All rows from last col



# Splitting the dataset into the Training set and Test set
# Test set performance should not differ too much from training sets if our model will adapt well
from sklearn.cross_validation import train_test_split
# Build training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)"""






