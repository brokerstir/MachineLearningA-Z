# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # -1 removes last col, makes matrix of features / independent variables
y = dataset.iloc[:, 1].values # 1 selects only 2nd col for dependent variable vector

# Splitting the dataset into the Training set and Test set, make test size a third 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling, comment out because not needed for simple linear regression, as library takes care of it.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # No params needed
regressor.fit(X_train, y_train) # Fits regressor object to the training sets, regressor is machine that learns on training set, can use this to make predictions on test set

# Predicting the Test set results
# Put predictions in vector
y_pred = regressor.predict(X_test)
# compare y_test (actual salaries) with y_pred
