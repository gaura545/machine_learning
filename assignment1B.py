# -*- coding: utf-8 -*-
"""
Created on Sat May 29 22:57:08 2021

@author: gs803
"""

import pandas as pd
dataset = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt')

X = dataset.iloc[:,[1,7]].values

from sklearn.cluster import KMeans
K_Means = KMeans(n_clusters=3, init = 'k-means++',random_state=1)
K_Means.fit_predict(X)

wcss = []
for i in range(1,15):
    K_Means = KMeans(n_clusters=i, init = 'k-means++',random_state=1)
    K_Means.fit_predict(X)
    wcss.append(K_Means.inertia_)
    print('i=',i,'wcss=',K_Means.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(1,15),wcss)
plt.title('elbow method')
plt.xlabel('no. of cluster')
plt.ylabel('wcss score')
plt.show()
    