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
