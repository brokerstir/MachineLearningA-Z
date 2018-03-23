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

# Build a linear model to compare with polynomial model

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# Fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2 # Adds a col of ind var squared
dataset$Level3 = dataset$Level^3
# dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset) # Type summary(poly_reg) in console to see results

