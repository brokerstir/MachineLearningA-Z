# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # Removes last col, keeps only independent vars
y = dataset.iloc[:, 4].values # Selects last col

# Encoding categorical data
# Encoding the Independent Variable State
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) # Select state col
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# Dependent variables don't need to be encoded
# 4th col of state is replaced with three cols at beginning with each col corresponding to each state

# Avoiding the Dummy Var Trap
# Library will take care of trap but added here as reminder
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Manual feature scaling not needed

