# Polynomial Regression

# Paste data preprocessing code
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # Selects 2 col, makes X matrix
y = dataset.iloc[:, 2].values # Selects 3 col, makes y vector

# Splitting the dataset into the Training set and Test set
# Commented out because data set is too small to split
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# Commented out becasue Linear Regression library will take care of feature scaling.
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# End data preprocessing

# Fit Linear and Polynomial Regressions and compare them

# Fitting Linear Regression to the dataset
# First create class
from sklearn.linear_model import LinearRegression
# Then create object of class
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
# First create class
from sklearn.preprocessing import PolynomialFeatures
# Then create object of class
# Var poly_reg will transform X matrix by adding exponentials of X as ind vars
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
# Transformation of new matrix of features complete

# Create new object
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

