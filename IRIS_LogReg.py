# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#IRIS Data Set Analysis

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset

df = pd.read_csv(r'C:\Users\Piyush\Desktop\IRIS_Data.csv')

#Initial analysis of dataset

df.head(2)
df.tail(2)
df.describe()
df.corr()
#As petal length and petal width seems to be highly corelated, removing 1 
#feature - petal width and copying the data to df1
df1 = df.iloc[:,:-2]
df1

#draw histogram and boxplot to check the data distribution and skewness

df['sepal_length'].hist(bins=10)
df['sepal_width'].hist(bins=10)
df['petal_length'].hist(bins=10)
df['petal_width'].hist(bins=10)

df.boxplot('sepal_length')
df.boxplot('sepal_width')
df.boxplot('petal_length')
df.boxplot('petal_width')

df.skew()

#No missing data and no label encoding is required
#split the data in train and test
x_train = df.iloc[:,:4]
y_train = df.iloc[:,4]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=20)

#select an algo to predict the value of unknown iris
#import the model/class
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#Instantiate the class - Create an object
LR = LogisticRegression()
#fit the training data
LR.fit(x_train,y_train)
y_pred = LR.predict(x_test)
y_pred

#Check accuracy

from sklearn.metrics import accuracy_score
Accuracy = accuracy_score(y_test, y_pred)
Accuracy












