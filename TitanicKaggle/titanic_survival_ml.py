# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:54:30 2018

@author: Broker Stir

For the Titanic Kaggle 
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('train_mod2.csv')

X = train_dataset.iloc[:, 2:6].values
y = train_dataset.iloc[:, 1].values

# Taking care of missing data
# Use mean age to sub for nan values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

# Encoding categorical data for gender
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() # Creates object
X[:, 1] = labelencoder_X.fit_transform(X[:, 1]) # Transform gender to categories
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray() # Encodes as categorical variables

# Avoid dummy var trap
X = X[:, 1:]

# NOTE: col 0 -> dummy var gender, col 1 -> PClass, col 2 -> Age, col 3 -> Fare
 

# Encoding categorical data for pClass
# Encoding the Independent Variable
labelencoder2_X = LabelEncoder() # Creates object
X[:, 1] = labelencoder2_X.fit_transform(X[:, 1]) # Transform Pclass to categories
# Transform Pclass as categorical
onehotencoder2 = OneHotEncoder(categorical_features = [1])
X = onehotencoder2.fit_transform(X).toarray() # Encodes as categorical variables

# Avoid dummy var trap
X = X[:, 1:]

# NOTE: col 0, 1  -> PClass dummy vars, col 2 -> dummy var gender, col 3 -> Age, col 4 -> Fare

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build Logistic Regression Model

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression # Import Class
classifier = LogisticRegression(random_state = 0) # Creates object of Class with using only one parameter
classifier.fit(X_train, y_train)

# Predicting the Test set results
# Create a vector of predictions with the test set
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
# Function that makes matrix of correct and incorrect predictions
from sklearn.metrics import confusion_matrix # Functions start with lower case
cm = confusion_matrix(y_test, y_pred)

# Build K-Nearest Neighbor Model

# Fitting K-NN to the Training set
# Import object, create object and fit object
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) # Number of neighbors base on euclidean distance
classifier_knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier_knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)

# Predicting the Test set results
y_pred_dt = classifier_dt.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)

