# Polynomial Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3] # Keep only last two cols
# Level is ind var and sal is dep var

# Commented out below because no training set nor test set will be created. The data set is already small.
# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling not need as library handles it, so commented out below
# training_set = scale(training_set)
# test_set = scale(test_set)

