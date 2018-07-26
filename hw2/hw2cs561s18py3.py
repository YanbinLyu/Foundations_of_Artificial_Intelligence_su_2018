import sys
import string
import re
import copy
import math
from sklearn.externals import joblib
import warnings

import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


f1 = open(sys.argv[1])
whole_thing = f1.read()
data = []
data.append(whole_thing)
f1.close()
with warnings.catch_warnings():
	warnings.simplefilter("ignore", category=UserWarning)
	clf = joblib.load('neuralNetwork.pickle')
# Preprocessing
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
stop_words = stopwords.words('english')
tokenize_list = word_tokenize(data[0])
tokenize_list = [x for x in tokenize_list if x not in stop_words]
for j in range(0, len(tokenize_list)):
	tokenize_list[j] = lemmatizer.lemmatize(stemmer.stem(tokenize_list[j]))
data[0] = ' '.join(tokenize_list)

result = clf.predict(data)
# Output file
f2 = open('output.txt', "w+")
if result == "positive":
	f2.write("1")
else:
	f2.write("0")
f2.write("\n")
f2.close()