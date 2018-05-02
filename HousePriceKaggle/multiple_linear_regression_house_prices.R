# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('train_mod2.csv')

# Encoding categorical data

dataset$LotConfig = factor(dataset$LotConfig,
                          levels = c('Corner', 'CulDSac', 'FR2', 'FR3', 'Inside'),
                          labels = c(1, 2, 3, 4, 5))



#dataset$MSZoning = factor(dataset$MSZoning,
                            #levels = c('A', 'C', 'FV', 'I', 'RH', 'RL', 'RM', 'RP'),
                            #labels = c(1, 2, 3, 4, 5, 6, 7, 8))

#dataset$BldgType = factor(dataset$BldgType,
                            #levels = c('1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'),
                            #labels = c(1, 2, 3, 4, 5))

dataset$HouseStyle = factor(dataset$HouseStyle,
                          levels = c('1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl'),
                          labels = c(1, 2, 3, 4, 5, 6, 7, 8))


#dataset$MSSubClass = factor(dataset$MSSubClass,
                              #levels = c('120', '150', '160', '180', '190', '20', '30', '40', '45', '50', '60', '70', '75', '80', '85', '90'),
                              #labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16))

dataset$Neighborhood = factor(dataset$Neighborhood,
                       levels = c('Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NPkVill', 'NWAmes', 'NAmes', 'NoRidge', 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'),
                       labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25))

#dataset$KitchenQual = factor(dataset$KitchenQual,
                            #levels = c('Ex', 'Fa', 'Gd', 'Po', 'TA'),
                            #labels = c(1, 2, 3, 4, 5))

#dataset$Functional = factor(dataset$Functional,
                            #levels = c('Maj1', 'Maj2', 'Min1', 'Min2', 'Mod', 'Sal', 'Sev', 'Typ'),
                            #labels = c(1, 2, 3, 4, 5, 6, 7, 8))

#dataset$SaleCondition = factor(dataset$SaleCondition  ,
                             #levels = c('Abnorml', 'AdjLand', 'Alloca', 'Family', 'Normal', 'Partial'),
                             #labels = c(1, 2, 3, 4, 5, 6))



# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$SalePrice, SplitRatio = 0.9)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Manual feature scaling not required

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set

# Building the optimal model using bacdward elmination
# Use regressor model but type out all ind vars

# Beauty of R is no need to create dummy vars, the factor function does this when encoding categorical data


# Complete backwared elimination, remove vars with high P values
regressor = lm(formula = SalePrice ~ OverallQual + OverallCond + GrLivArea + FullBath + HalfBath + GarageArea + LotArea + HouseStyle + LotConfig + Neighborhood,
               data = dataset) 

summary(regressor)

# Predicting the Test set resultsy
y_pred = predict(regressor, newdata = test_set)

write.csv(y_pred, file = "y_pred1.csv")


