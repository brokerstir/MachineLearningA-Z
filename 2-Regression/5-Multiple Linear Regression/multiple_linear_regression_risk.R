# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Manual feature scaling not required

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set

# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
# Short Version
regressor = lm(formula = Profit ~ .,
               data = training_set)


# Profit is linear combination of ind vars
# Type sumarry(regressor) in console
# Notice dummy vars were implemented automatically

# Lower p value means higher sinigicance of ind vars.

# In this example, the only strong predictor is RD spend
# So convert this to simple linear
# regressor = lm(formula = Profit ~ R.D.Spend, data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Type y_pred in console to look at it



