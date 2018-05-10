# Apriori

# Data Preprocessing
#install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE) # header false argument shows the first row not as a header but as a row of observations

# The arules package takes a sparse matrix as input so dataset must be transformed to a sparse matrix. This will have 120 cols for each product. The rows will still transactions but will have a 0 or 1 depending if product was buy or no buy
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE) # last argument removes any duplicats in each list of transactions


summary(dataset) # gives details about sparse matrix, i.e. density is ratio of 1 values, distribution is number of transactions that have 1 product, 2 products, etc.

itemFrequencyPlot(dataset, topN = 10) # Frequency plot of top 10 products, Use this chart to choose good value for support.

# Coefficient is price of product, support should be set to items purchase frequently
# support above product bought 3 times per day that's 21 per week / 7500 weekl transactions rounds to  .003 .... note, this is arbitrary, can be changed
# start with a default confidence and decrease step by step until there are good associations, high confidence will give us obvious rules where too small will give silly rules / 0.8 is default for now. But 0.8 returned no rules, so to high, nothing matches 80% of time, try 0.4
# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10]) #gives first ten rules sorted by life, or relevance

# Chocolat and herb pepper is not a good association, but in the rules because it has hight support, so we should change confidence, we don't want to change support in this case, so lower confidence to 0.2

# Now, perhaps change the support to products bought 4 times a day