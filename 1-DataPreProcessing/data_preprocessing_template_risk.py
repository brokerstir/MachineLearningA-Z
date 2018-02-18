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

# Take care of missing data
from sklearn.preprocessing import Imputer # Class
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #Object
imputer = imputer.fit(X[:, 1:3]) # Use instance in 2nd and 3rd cols
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Replace missing data with mean

