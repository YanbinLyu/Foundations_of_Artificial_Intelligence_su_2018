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
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Load datasets
datasets = pd.read_csv('reviews.csv', sep=', ', delimiter='|')
x_test = []
y_test1 = []
# Divide into training data and test data
print("Loaded 1")
# Preprocessing
lemmatizer = WordNetLemmatizer()
stemmer = LancasterStemmer()
stop_words = stopwords.words('english')
for i in range(0, len(datasets)):
	if i%5 == 4:
		tokenize_list = word_tokenize(datasets['text'][i])
		tokenize_list = [x for x in tokenize_list if x not in stop_words]
		for j in range(0, len(tokenize_list)):
			tokenize_list[j] = lemmatizer.lemmatize(stemmer.stem(tokenize_list[j]))
		text = ' '.join(tokenize_list)
		x_test.append(text)
		y_test1.append(datasets['label'][i])
print("Loaded 2")
print()

# Load clf
text_clf1 = joblib.load('decisionTree.pickle')
text_clf2 = joblib.load('neuralNetwork.pickle')
text_clf3 = joblib.load('naiveBayes.pickle')

# Run the classifier and get some data

# Accuracies 
predictedResult1 = text_clf1.predict(x_test)
print('The accuracy of Decision tree is {0:0.2f}'.format(np.mean(predictedResult1 == y_test1)))

predictedResult2 = text_clf2.predict(x_test)
print('The accuracy of Neural network is {0:0.2f}'.format(np.mean(predictedResult2 == y_test1)))

predictedResult3 = text_clf3.predict(x_test)
print('The accuracy of Naive bayes is {0:0.2f}'.format(np.mean(predictedResult3 == y_test1)))

# Compute the average precision score and f1 score 
# Have to change the data into binary first
predicted1 = [1 if x == 'positive' else 0 for x in predictedResult1]
predicted2 = [1 if x == 'positive' else 0 for x in predictedResult2]
predicted3 = [1 if x == 'positive' else 0 for x in predictedResult3]

y_test = [1 if x == 'positive' else 0 for x in y_test1]


# Decision Tree
# average_precision1 = average_precision_score(y_test, predicted1)
# print('Decision Tree: Average precision-recall score of : {0:0.2f}'.format(
#       average_precision1))


# fpr1, tpr1, threshold1 = roc_curve(y_test, predicted1)
# roc_auc1 = auc(fpr1, tpr1)

# plt.subplot(3, 1, 1)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % roc_auc1)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')

# fpr2, tpr2, threshold2 = roc_curve(y_test, predicted2)
# roc_auc2 = auc(fpr2, tpr2)

# plt.subplot(3, 1, 2)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % roc_auc2)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')

# fpr3, tpr3, threshold3 = roc_curve(y_test, predicted3)
# roc_auc3 = auc(fpr3, tpr3)

# plt.subplot(3, 1, 3)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr3, tpr3, 'b', label = 'AUC = %0.2f' % roc_auc3)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()




# PRECISION/RECALL
# precision1, recall1, _ = precision_recall_curve(y_test, predicted1)

# plt.subplot(3, 1, 1)
# plt.step(recall1, precision1, color='b', alpha=0.2,
#          where='post')
# plt.fill_between(recall1, precision1, step='post', alpha=0.2,
#                  color='b')

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('Decision Tree 2-class Precision-Recall curve: AP={0:0.2f}'.format(
#           average_precision1))

f1_score1 = f1_score(y_test, predicted1) 
print('Decision Tree: f1 score: {0:0.2f}'.format(
      f1_score1))

# # Neural Network
# average_precision2 = average_precision_score(y_test, predicted2)
# print('Neural Network: Average precision-recall score of : {0:0.2f}'.format(
#       average_precision2))

# precision2, recall2, _ = precision_recall_curve(y_test, predicted2)

# plt.subplot(3, 1, 2)
# plt.step(recall2, precision2, color='g', alpha=0.2,
#          where='post')
# plt.fill_between(recall2, precision2, step='post', alpha=0.2,
#                  color='b')

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('Neural Network 2-class Precision-Recall curve: AP={0:0.2f}'.format(
#           average_precision2))
f1_score2 = f1_score(y_test, predicted2) 
print('Neural Network: f1 score: {0:0.2f}'.format(
      f1_score2))

# # Naive Bayes
# average_precision3 = average_precision_score(y_test, predicted3)
# print('Naive Bayes: Average precision-recall score of : {0:0.2f}'.format(
#       average_precision3))
# precision3, recall3, _ = precision_recall_curve(y_test, predicted2)

# plt.subplot(3, 1, 3)
# plt.step(recall3, precision3, color='r', alpha=0.2,
#          where='post')
# plt.fill_between(recall3, precision3, step='post', alpha=0.2,
#                  color='b')

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('Naive Bayes 2-class Precision-Recall curve: AP={0:0.2f}'.format(
#           average_precision3))
f1_score3 = f1_score(y_test, predicted3) 
print('Naive Bayes: f1 score: {0:0.2f}'.format(
      f1_score3))
# plt.show()

