import learn as learn
import numpy as np
import pandas as pd
import pip
from pip.commands import install, show
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import os
import csv
import sys
from xml.dom.minidom import Document
from sklearn.metrics.classification import classification_report, accuracy_score, confusion_matrix
from sklearn import tree
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import random
import dataframe as df

src = sys.argv[1]+'profile/'
filename = 'profile.csv'

# Predicting gender by using "page-likes"
profile_csv = pd.read_csv("C:/temp/tcss555/training/profile/profile.csv")
relation_csv = pd.read_csv("C:/temp/tcss555/training/relation/relation.csv")


def preparing_dataset(profile_data, relation_data):
    merge_PR_train_csv = pd.merge(profile_data, relation_data, on='userid', how='inner')
    return merge_PR_train_csv

features = preparing_dataset(profile_csv,relation_csv)

#print(features)

# Splitting the data into 300 training instances and 104 test instances
n = 1500
all_Ids = np.arange(len(features))
import random
test_Ids = all_Ids[0:n]
train_Ids = all_Ids[n:]
data_test = features.loc[test_Ids, :]
data_train = features.loc[train_Ids, :]

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(data_train['transcripts'])
y_train = data_train['gender']
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Testing the Naive Bayes model
X_test = count_vect.transform(data_test['transcripts'])
y_test = data_test['gender']
y_predicted = clf.predict(X_test)

# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))
classes = ['Male','Female']
cnf_matrix = confusion_matrix(y_test,y_predicted,labels=classes)
print("Confusion matrix:")
print(cnf_matrix)
