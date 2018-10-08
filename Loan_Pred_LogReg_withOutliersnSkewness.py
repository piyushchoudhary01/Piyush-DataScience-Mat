# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:12:49 2018

@author: Piyush
"""

#Loan_Prediction Dataset analysis using Logistic Regression without reducing the skewness of
#features or scaling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
df = pd.read_csv(r'C:\Users\Piyush\Desktop\Data Science Material and Links\Data Sets to Practice\train.csv')
df1=df.copy()
#Analyze the data for missing values, corelation etc
df.head(5)
df.tail(5)
desc = df.describe()
corr = df.corr()

#Check for data distribution and skewness
df['ApplicantIncome'].hist(bins=10)
df['CoapplicantIncome'].hist(bins=10)
df['LoanAmount'].hist(bins=10)
df['Loan_Amount_Term'].hist(bins=10)
df['Credit_History'].hist(bins=10)

df.boxplot('ApplicantIncome')
df.boxplot('CoapplicantIncome')
df.boxplot('LoanAmount')
df.skew()

#fill the missing values
df1['LoanAmount'].fillna(df1['LoanAmount'].mean(),inplace=True)
df1['Loan_Amount_Term'].fillna(df1['Loan_Amount_Term'].mean(),inplace=True)
df1['Credit_History'].fillna(df1['Credit_History'].mean(),inplace=True)
desc1=df1.describe()

#Label Encoding of Loan Status

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
cat_var = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
for i in cat_var:
    df1[i]=enc.fit_transform(df1[i].astype(str))
#Could use Impute method also to fill NaN in place of missing values

x_train = df1.iloc[:,1:-1]
y_train=df1.iloc[:,-1]


#split the test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.25,random_state=33)
#Import Logistic Regrssion class

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)

#CHeck for accuracy
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc

#Accuracy = .80519480




