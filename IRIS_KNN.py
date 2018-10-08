# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:54:53 2018

@author: Piyush
"""

#IRIS Data Set Analysis using KNN Algo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Piyush\Desktop\IRIS_Data.csv')

x_train = df.iloc[:,:-1]
y_train = df.iloc[:,4]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.25,random_state=40)

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=2)
KNN.fit(x_train,y_train)
y_pred = KNN.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
acc
