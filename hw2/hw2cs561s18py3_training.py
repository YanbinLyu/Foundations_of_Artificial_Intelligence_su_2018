import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
# from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import numpy as np

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


# Load datasets
datasets = pd.read_csv('reviews.csv', sep=', ', delimiter='|')
y_train = []
x_train = []
x_test = []
y_test = []
# Divide into training data and test data
print("Loaded 1")
# Preprocessing
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
stop_words = stopwords.words('english')
for i in range(0, len(datasets)):
	tokenize_list = word_tokenize(datasets['text'][i])
	tokenize_list = [x for x in tokenize_list if x not in stop_words]
	for j in range(0, len(tokenize_list)):
		tokenize_list[j] = lemmatizer.lemmatize(stemmer.stem(tokenize_list[j]))
	text = ' '.join(tokenize_list)
	if i%5 == 4:
		x_test.append(text)
		y_test.append(datasets['label'][i])
	else:
		y_train.append(datasets['label'][i])
		x_train.append(text)
print("Loaded 2")
print()
text_clf1 = Pipeline([('vect', HashingVectorizer(n_features=2000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', tree.DecisionTreeClassifier()),
])
text_clf1.fit(x_train, y_train) 
predicted1 = text_clf1.predict(x_test)
print("Decision Tree!!!")
joblib.dump(text_clf1, 'decisionTree.pickle')
print(np.mean(predicted1 == y_test)) 

text_clf2 = Pipeline([('vect', HashingVectorizer(n_features=2000)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MLPClassifier(hidden_layer_sizes=(500,))),
])
text_clf2.fit(x_train, y_train) 
predicted2 = text_clf2.predict(x_test)
print("Neural Network!!!")
joblib.dump(text_clf2, 'neuralNetwork.pickle')
print(np.mean(predicted2 == y_test)) 


text_clf3 = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])
text_clf3.fit(x_train, y_train) 
predicted3 = text_clf3.predict(x_test)
print("Naive Bayes!!!")
joblib.dump(text_clf3, 'naiveBayes.pickle')
print(np.mean(predicted3 == y_test))





