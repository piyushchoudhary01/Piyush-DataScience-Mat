# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 12:24:59 2018

@author: Piyush
"""

#Loan Data Analysis after Using Log for Applicant Income

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Piyush\Desktop\Data Science Material and Links\Data Sets to Practice\train.csv')
df3=df.copy()
df4=df.copy()
df3['ApplicantIncome'].hist(bins=10)
df3['CoapplicantIncome'].hist(bins=10)
df3['LoanAmount'].hist(bins=10)

df3.skew()

df4['ApplicantIncome']=np.log(df3['ApplicantIncome'])
df4['ApplicantIncome'].hist(bins=10)
df4.skew()

desc1=df4.describe()
df4['ApplicantIncome'].fillna(df4['ApplicantIncome'].mean())
df4['CoapplicantIncome'].fillna(df4['CoapplicantIncome'].mean())

df4['Loan_Amount_Term'].fillna(df4['Loan_Amount_Term'].mean(),inplace=True)
df4['Credit_History'].fillna(df4['Credit_History'].mean(),inplace=True)
df4['LoanAmount'].fillna(df4['LoanAmount'].mean(),inplace=True)


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
cat_var = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
for i in cat_var:
    df4[i] = enc.fit_transform(df4[i].astype(str))
   
x_train = df4.iloc[:,1:-1]
y_train = df4.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.25,random_state=22)

from sklearn.linear_model import LogisticRegression
LR1=LogisticRegression()
LR1.fit(x_train,y_train)
y_pred=LR1.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
acc

#Accuracy = 0.83766


