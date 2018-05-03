# Multiple Linear Regression

# Importing the dataset
traindataset = read.csv('train_mod2.csv')
testdataset = read.csv('test_mod2.csv')

# Encoding categorical data

traindataset$BsmtFinType1 = factor(traindataset$BsmtFinType1,
                              levels = c('ALQ', 'BLQ', 'GLQ', 'LwQ', 'NA', 'Rec', 'Unf'),
                              labels = c(1, 2, 3, 4, 5, 6, 7))

testdataset$BsmtFinType1 = factor(testdataset$BsmtFinType1,
                                   levels = c('ALQ', 'BLQ', 'GLQ', 'LwQ', 'NA', 'Rec', 'Unf'),
                                   labels = c(1, 2, 3, 4, 5, 6, 7))


traindataset$BsmtQual = factor(traindataset$BsmtQual,
                          levels = c('Ex', 'Fa', 'Gd', 'NA', 'Po', 'TA'),
                          labels = c(1, 2, 3, 4, 5, 6))

testdataset$BsmtQual = factor(testdataset$BsmtQual,
                               levels = c('Ex', 'Fa', 'Gd', 'NA', 'Po', 'TA'),
                               labels = c(1, 2, 3, 4, 5, 6))



traindataset$ExterQual = factor(traindataset$ExterQual,
                           levels = c('Ex', 'Fa', 'Gd', 'Po', 'TA'),
                           labels = c(1, 2, 3, 4, 5))

testdataset$ExterQual = factor(testdataset$ExterQual,
                                levels = c('Ex', 'Fa', 'Gd', 'Po', 'TA'),
                                labels = c(1, 2, 3, 4, 5))




traindataset$LotConfig = factor(traindataset$LotConfig,
                          levels = c('Corner', 'CulDSac', 'FR2', 'FR3', 'Inside'),
                          labels = c(1, 2, 3, 4, 5))

testdataset$LotConfig = factor(testdataset$LotConfig,
                                levels = c('Corner', 'CulDSac', 'FR2', 'FR3', 'Inside'),
                                labels = c(1, 2, 3, 4, 5))





traindataset$HouseStyle = factor(traindataset$HouseStyle,
                          levels = c('1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl'),
                          labels = c(1, 2, 3, 4, 5, 6, 7, 8))

testdataset$HouseStyle = factor(testdataset$HouseStyle,
                                 levels = c('1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl'),
                                 labels = c(1, 2, 3, 4, 5, 6, 7, 8))




traindataset$Neighborhood = factor(traindataset$Neighborhood,
                       levels = c('Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NPkVill', 'NWAmes', 'NAmes', 'NoRidge', 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'),
                       labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25))

testdataset$Neighborhood = factor(testdataset$Neighborhood,
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
'''library(caTools)
set.seed(123)
split = sample.split(dataset$SalePrice, SplitRatio = 0.9)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)'''

# Manual feature scaling not required

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set

# Building the optimal model using bacdward elmination
# Use regressor model but type out all ind vars

# Beauty of R is no need to create dummy vars, the factor function does this when encoding categorical data


# Complete backwared elimination, remove vars with high P values
regressor = lm(formula = SalePrice ~ OverallQual + OverallCond + GrLivArea + FullBath + HalfBath + GarageArea + LotArea + HouseStyle + LotConfig + Neighborhood + ExterQual + BsmtQual + BsmtFinType1,
               data = traindataset) 

summary(regressor)

# Predicting the Test set resultsy
y_pred = predict(regressor, newdata = testdataset)

write.csv(y_pred, file = "y_pred_submission2.csv")


