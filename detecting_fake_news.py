# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 00:12:32 2021

@author: Zeinab Khattab
"""
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Now, let’s read the data into a DataFrame, and get the shape of the data and the first 5 records.
df=pd.read_csv('E:\\DataScience\\Projects\\detecting_fake_news_1\\news.csv')
df.shape
df.head()

#And get the labels from the DataFrame.
labels=df.label
labels.head()

#Split the dataset into training and testing sets.
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#DataFlair_Initialize a TfidVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#DataFlair-Initialize a passiveAggressiveClassifier

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

confusion_matrix(y_test,y_pred, labels=['FAKE', 'REAL'])


