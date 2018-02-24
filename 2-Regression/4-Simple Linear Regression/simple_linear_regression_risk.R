# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3) # 2/3 in Training Set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling is taken care of by linear regression package in R
# training_set = scale(training_set)
# test_set = scale(test_set)

# data preprocessing complete

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

# Type summary(regressor) in console and notice *** for strength of dependence
# Low P value indicates independent variable is highly significant

# Predicting the Test set Results
y_pred = predict(regressor, newdata = test_set)
# After execution, type y_pred in console to see values