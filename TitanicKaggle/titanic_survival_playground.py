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
train_dataset = pd.read_csv('train_mod1.csv')
test_dataset = pd.read_csv('test_mod1.csv')

X_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, 6].values

X_test = test_dataset.iloc[:, :].values

# Taking care of missing data
# Use mean age to sub for nan values
from sklearn.preprocessing import Imputer
imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean',  axis = 0)
imputer1 = imputer1.fit(X_train[:, 3:4])
X_train[:, 3:4] = imputer1.transform(X_train[:, 3:4])

imputer2 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer2 = imputer2.fit(X_test[:, 3:4])
X_test[:, 3:4] = imputer2.transform(X_test[:, 3:4])

imputer3 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer3 = imputer3.fit(X_test[:, 5:])
X_test[:, 5:] = imputer3.transform(X_test[:, 5:])


X_id = X_test[:, 0] # Array of id
X_train = X_train[:, 1:] # Remove col of id from X
X_test = X_test[:, 1:] # Remove col of id from X

#X_train[:, 2:3] = X_train[:, 2:3].round(decimals=2)

# Encoding categorical data for gender
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder() # Creates object
X_train[:, 1] = labelencoder_X1.fit_transform(X_train[:, 1]) # Transform gender to categories
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X2 = LabelEncoder() # Creates object
X_test[:, 1] = labelencoder_X2.fit_transform(X_test[:, 1]) # Transform gender to categories




onehotencoder1 = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder1.fit_transform(X_train).toarray() # Encodes as categorical variables

#X_train[:, 3:4] = X_train[:, 3:4].round(decimals=2)

onehotencoder2 = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder2.fit_transform(X_test).toarray() #

# Avoid dummy var trap
X_train = X_train[:, 1:]
# Avoid dummy var trap
X_test = X_test[:, 1:]

# NOTE: 
#col 0 -> dummy var gender
#col 1 -> PClass
#col 2 -> Age
#col 3 -> Sib
#col 4 -> Parch
#col 5 -> Fare
 
# Encoding categorical data for pClass
# Encoding the Independent Variable
labelencoder2_X1 = LabelEncoder() # Creates object
X_train[:, 1] = labelencoder2_X1.fit_transform(X_train[:, 1]) # Transform Pclass to categories

labelencoder2_X2 = LabelEncoder() # Creates object
X_test[:, 1] = labelencoder2_X2.fit_transform(X_test[:, 1]) # Transform Pclass to categories

# Transform Pclass as categorical
onehotencoder3 = OneHotEncoder(categorical_features = [1])
X_train = onehotencoder3.fit_transform(X_train).toarray() # Encodes as categorical variables
onehotencoder4 = OneHotEncoder(categorical_features = [1])
X_test = onehotencoder4.fit_transform(X_test).toarray() # Encodes as categorical variables

# Avoid dummy var trap
X_train = X_train[:, 1:]
# Avoid dummy var trap
X_test = X_test[:, 1:]

# NOTE: 
#col 0, 1-> dummy var Plass
#col 2 -> dummy var Gender
#col 3 -> Age
#col 4 -> Sib
#col 5 -> Parch
#col 6 -> Fare

# Encoding categorical data for Sib
# Encoding the Independent Variable
labelencoder3_X1 = LabelEncoder() # Creates object
X_train[:, 4] = labelencoder3_X1.fit_transform(X_train[:, 4]) # Transform Pclass to categories
labelencoder3_X2 = LabelEncoder() # Creates object
X_test[:, 4] = labelencoder3_X2.fit_transform(X_test[:, 4]) # Transform Pclass to categories
# Transform Pclass as categorical
onehotencoder5 = OneHotEncoder(categorical_features = [4])
X_train = onehotencoder5.fit_transform(X_train).toarray() # Encodes as categorical variables
onehotencoder6 = OneHotEncoder(categorical_features = [4])
X_test = onehotencoder6.fit_transform(X_test).toarray() # Encodes as categorical variables

# Avoid dummy var trap
X_train = X_train[:, 1:]
# Avoid dummy var trap
X_test = X_test[:, 1:]


# NOTE: 
#col 0-5 -> dummy var Sib
#col 6, 7-> dummy var Plass
#col 8 -> dummy var Gender
#col 9 -> Age
#col 10 -> Parch
#col 11 -> Fare



# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)

# Splitting the dataset into the Training set and Test set
# No Data Split, Already Split

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

np.savetxt("predicts.csv", y_pred, delimiter=",")

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

