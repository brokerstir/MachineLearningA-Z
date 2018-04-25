# Natural Language Processing

# STEP 1

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) # Tab is set as delimiter and double quotes will be ignored

# STEP 2

'''
# Cleaning the texts by individual line
import re
import nltk
nltk.download('stopwords') # This is a list of stop word that we want to remove from the reviews
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) # Removes everything but letters, replaces with a space, applied to first review for testing

# STEP 3 & 4 & 5

# Make all letters lower case
review = review.lower()
# Split review into list of words
review = review.split()
# Create object of stemmer class
ps = PorterStemmer()
# Loop through words in review and keep words if not in stopwords
# review = [word for word in review if not word in set(stopwords.words('english'))]
# Use stemmer object
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
# Join words back form list to string
review = ' '.join(review)
'''


# Cleaning the texts usinga for loop
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
# Bag of words model creates a column for each word and a row for each review. A cell gets a 1 or 0 depending if the word appears in the review. Most cells will be 0. This is called sparsity or a sparse matrix.s
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # Max features parameters filters out less frequent words, and limits number of word columns
X = cv.fit_transform(corpus).toarray() # Creates sparse matrix of features
y = dataset.iloc[:, 1].values # Creates vector for dependent var

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) # Chose 20% for test size to have more data for training...since data set was larage

# NOTE: Feature scaling is not needed since our matrix is almost all 1 and 0

# Naive Bayes, Decision Tree, and Random Forest are common models used for NLP. In this example we will use Naive Bayes
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy = (55+91)/200 # .73 accuracy is fair considering the training size of 800. A bigger training size could improve accuracy