# Multiple Linear Regression

# STEP 1

# Importing the dataset
gradesdataset = read.csv('50_Grades_Data.csv')

# Encoding categorical data
gradesdataset$Subject = factor(gradesdataset$Subject,
                               levels = c('English', 'Math'),
                               labels = c(1, 2))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(gradesdataset$Subject, SplitRatio = 0.8)
training_set = subset(gradesdataset, split == TRUE)
test_set = subset(gradesdataset, split == FALSE)

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

# Type y_pred in console to look at it, and compare tith test_set

# Building the optimal model using bacdward elmination
# Use regressor model but type out all ind vars
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset) # on whole data set for complete info
# Beauty of R is no need to create dummy vars, the factor function does this when encoding categorical data

summary(regressor) # Returns data to search for vars with P value above signif level

# Complete backwared elimination, remove vars with high P values, first to remove is State
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset) 
summary(regressor)

# Next, remove Administration
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset) 
summary(regressor)

# Next, remove Marketing.Spend
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset) 
summary(regressor)




