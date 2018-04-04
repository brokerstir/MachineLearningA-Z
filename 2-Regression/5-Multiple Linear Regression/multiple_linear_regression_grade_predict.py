# Multiple Linear Regression

# STEP 1

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
gradesdataset = pd.read_csv('50_Grades_Data.csv')
X = gradesdataset.iloc[:, :-1].values # Removes last col, keeps only independent vars
# Since X will be of type object, input X into console to vew the matrix values
y = gradesdataset.iloc[:, 3].values # Selects last col

# Encoding categorical data
# Encoding the Independent Variable State
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2]) # Select state col
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()
# Dependent variables don't need to be encoded


# Avoiding the Dummy Var Trap
# Library will take care of trap but added here as reminder
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Manual feature scaling not needed

# STEP 2

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# STEP 3

# Predicting the Test set results
y_pred = regressor.predict(X_test)

######################################################

# Backward Elimination
# STEP 1

# The goal is to find a team of independent variables that have the most statistical significance. 



# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# Create column of 1s to associate with constant B zero, so the coloumn of 1s can be thought of as X zero equal to 1
X =  np.append(arr = np.ones((50,1)).astype(int), values = X, axis =1)

# Create matrix only with independent vars that have hight impact
# Start with matrix of features of all ind vars
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Create new regressor object
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Look for predictor with highest p value
regressor_OLS.summary()

# See in console that X2 has p value highest
# Remove X2
X_opt = X[:, [0, 1, 3, 4, 5]]
# Create new regressor object
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Look for predictor with highest p value
regressor_OLS.summary()


X_opt = X[:, [0, 3, 4, 5]]
# Create new regressor object
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Look for predictor with highest p value
regressor_OLS.summary()


X_opt = X[:, [0, 3, 5]]
# Create new regressor object
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Look for predictor with highest p value
regressor_OLS.summary()


X_opt = X[:, [0, 3]]
# Create new regressor object
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# Look for predictor with highest p value
regressor_OLS.summary()
