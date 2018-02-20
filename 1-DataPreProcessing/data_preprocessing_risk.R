# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# install commented out after it's executed once

# Call library
library(caTools)
set.seed(123) # choose any number
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# Returns true if observations goes to training set

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling allows machine learning models to converge quickly
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3]) # [, 2:3] selecting columns to apply scaling
