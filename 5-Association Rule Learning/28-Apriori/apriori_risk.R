# Apriori

# Data Preprocessing
#install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE) # header false argument shows the first row not as a header but as a row of observations

# The arules package takes a sparse matrix as input so dataset must be transformed to a sparse matrix. This will have 120 cols for each product. The rows will still transactions but will have a 0 or 1 depending if product was buy or no buy
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE) # last argument removes any duplicats in each list of transactions


summary(dataset) # gives details about sparse matrix, i.e. density is ratio of 1 values, distribution is number of transactions that have 1 product, 2 products, etc.

itemFrequencyPlot(dataset, topN = 10) # Frequency plot of top 10 products, Use this chart to choose good value for support.

# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])