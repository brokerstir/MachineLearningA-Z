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

# Country and Purchased Cols are categorical variables
# Need to encode text of these variables in to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create first object of this class
labelencoder_X = LabelEncoder() # Creates object
# Use object on col
X[:, 0] = labelencoder_X.fit_transform(X [:, 0]) # fits object to first column and returns it encoded
# Prevent equations from thinking there is relational order to countries, so create dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0]) #Call Class
# Note, categorical_feautres = 0 applies to 1st col index

# Fit to the matrix
X = onehotencoder.fit_transform(X).toarray()

# Encode last col for purchased, this is dependent variable so it's known that it's not relational order, only label enoder needed
labelencoder_y = LabelEncoder() # Creates object
# Use object on col
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
# Test set performance should not differ too much from training sets if our model will adapt well
from sklearn.cross_validation import train_test_split
# Build training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
