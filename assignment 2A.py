# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:22:10 2021

@author: gs803
"""

import pandas as pd
dataset = pd.read_csv('https://raw.githubusercontent.com/tranghth-lux/data-science-complete-tutorial/master/Data/HR_comma_sep.csv.txt')

dataset.isnull().values.any()

dataset.corr()

salary_map = {'low':-1,'medium':0,'high':1}
dataset['salary'] = dataset['salary'].map(salary_map)

del dataset['sales']

X = dataset.drop(columns = ['left'])
y = dataset['left']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

print(classifier.score(X_test,y_test))