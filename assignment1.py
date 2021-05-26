# -*- coding: utf-8 -*-
"""
Created on Thu May 27 02:24:08 2021

@author: gs803
"""

import pandas as pd
dataset = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt')
print(dataset)

X = dataset.iloc[:,1:7].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.neighbors import KNeighborsRegressor
K_NN = KNeighborsRegressor()
K_NN.fit(X_train,y_train)

y_predict = K_NN.predict(X_test)
print(y_predict)
print(K_NN.score(X_test,y_test))
