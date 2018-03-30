# Simple Linear Regression

# STEP 1

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
gradesdata = pd.read_csv('Grades_Data.csv')
X = gradesdata.iloc[:, :-1].values # -1 removes last col, makes matrix of features / independent variables
y = gradesdata.iloc[:, 1].values # 1 selects only 2nd col for dependent variable vector

# Splitting the dataset into the Training set and Test set, make test size a third 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# STEP 2

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression # Import specific libray to make the regressor object of LinearRegression class
regressor = LinearRegression() # No params needed
regressor.fit(X_train, y_train) # Method that fits regressor object to the training sets, regressor is machine that learns on training set, can use this to make predictions on test set, regressor has learned correllations.

# STEP 3

# Predicting the Test set results
# Put predictions in vector, a vector of grade predictions based on test set
y_pred = regressor.predict(X_test) # Use predict method on test set

# compare y_test (actual salaries) with y_pred

# STEP 4

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red') # Actual data (observation points) of training set plotted using scatter function
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Plot of predicted values on training set, regression line using plot function
# Notice above we plot predicted values of training set here
# Real values will bue red points, The blue line is the prediction regressor
plt.title('Grades vs Minutes of Study (Training set)')
plt.xlabel('Minutes of Study')
plt.ylabel('Grade')
plt.show()

# Visualising the Test set results (against same unique regression line built on training set)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# Note we keep the train set for the regressor line because we use the training set for the maching to learn the correlations
plt.title('Grades vs Minutes of Study (Test set)')
plt.xlabel('Minutes of Study')
plt.ylabel('Grade')
plt.show()