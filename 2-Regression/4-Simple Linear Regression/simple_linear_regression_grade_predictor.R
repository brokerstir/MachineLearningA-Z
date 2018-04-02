# Simple Linear Regression

# STEP 1

# Importing the dataset
gradesdataset = read.csv('Grades_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(gradesdataset$Grade, SplitRatio = 2/3) # 2/3 in Training Set
training_set = subset(gradesdataset, split == TRUE)
test_set = subset(gradesdataset, split == FALSE)

# STEP 2

# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Grade ~ MinutesStudy, # f1 calls help for lm function
               data = training_set)

# Type summary(regressor) in console and notice *** for strength of dependence
# Low P value indicates independent variable is highly significant

# STEP 3

# Predicting the Test set Results
y_pred = predict(regressor, newdata = test_set)
# After execution, type y_pred in console to see values

# STEP 4

# Visualising the Training set results
# install.packages('ggplot2') // Commented out aftr install

# Call library
# library(ggplot2)

# Use functions to plot observation points, regression line, label axis. Points plotted with real observation data, but line with y predicted values
ggplot() +
  geom_point(aes(x = training_set$MinutesStudy, y = training_set$Grade),
             colour = 'red') +
  geom_line(aes(x = training_set$MinutesStudy, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Grade vs Minutes Studied (Training set)') +
  xlab('Minutes Studied') +
  ylab('Grade')


# Visualising the Test set results. Keep regression line oon training set, because it's a already trained on training set, using test set would obtain same line with new points.
ggplot() +
  geom_point(aes(x = test_set$MinutesStudy, y = test_set$Grade),
             colour = 'red') +
  geom_line(aes(x = training_set$MinutesStudy, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Grade vs Minutes Studied (Test set)') +
  xlab('Minutes Studied') +
  ylab('Grade')
