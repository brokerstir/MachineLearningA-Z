# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')

# No need to create matrices or independent and dependent vars

# Take care of missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age) # Replaces missing age values with mean

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary) # Replaces missing salary values with mean

# Dummy variables for R is easy and done with vectors
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3)) # f1 for help
# Note, labels country as 1, 2, 3  and changes dataset file

dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No','Yes'),
                         labels = c(0, 1)) # f1 for help
# Note, labels No Yes as 0 1
