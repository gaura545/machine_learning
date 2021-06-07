# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 23:08:48 2021

@author: gs803
"""

import pandas as pd
dataset = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt')

dataset.isnull().values.any()

p = dataset.corr()

X = dataset.iloc[:,1:7]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
print(linear_model.score(X_test,y_test))