# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 16:04:38 2018

@author: Piyush
"""

#Loan Data Analysis using Decision Tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Piyush\Desktop\Data Science Material and Links\Data Sets to Practice\train.csv')
df1=df.copy()
df2=df.copy()

df2['ApplicantIncome']=np.log(df1['ApplicantIncome'])
df2['Loan_Amount_Term'].fillna(df2['Loan_Amount_Term'].mean(),inplace=True)
df2['Credit_History'].fillna(df2['Credit_History'].mean(),inplace=True)
df2['LoanAmount'].fillna(df2['LoanAmount'].mean(),inplace=True)

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
cat_var = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
for i in cat_var:
    df2[i] = enc.fit_transform(df2[i].astype(str))
   
x_train = df2.iloc[:,1:-1]
y_train = df2.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.25,random_state=30)

#from sklearn.tree import DecisionTreeClassifier
#DTC=DecisionTreeClassifier(criterion='gini',max_depth=None,min_samples_split=10,splitter='best')
#DTC.fit(x_train,y_train)
#y_pred=DTC.predict(x_test)

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=10,criterion='gini')
RFC.fit(x_train,y_train)
y_pred = RFC.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
acc

#Decision Tree Accuracy = .70779
#Random Forest Accuracy = .73377