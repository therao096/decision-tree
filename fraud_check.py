# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:36:25 2020

@author: Varun
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\decision tress\\Fraud_check.csv")
le=LabelEncoder()
data['Undergrad']=le.fit_transform(data['Undergrad'])
data['Marital.Status']=le.fit_transform(data['Marital.Status'])
data['Urban']=le.fit_transform(data['Urban'])

data
data.rename(columns={'Taxable.Income':'TaxableIncome'},inplace=True)
data
data['TaxableIncome'].max()
data['TaxableIncome']=pd.cut(data.TaxableIncome, bins=[0,30000,99619], labels=['Risky','Good'])
data=data[['Undergrad','Marital.Status','City.Population','Work.Experience','Urban','TaxableIncome']]
data.shape
data['TaxableIncome']=le.fit_transform(data['TaxableIncome'])
##pd.Series(TaxableIncome).value_counts()
colnames=list(data.columns)

predictors=colnames[0:5]
target=colnames[5]
train,test= train_test_split(data,test_size=0.3,random_state=0)
model=DecisionTreeClassifier(criterion="entropy")
model.fit(train[predictors],train[target])

preds=model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)
np.mean(preds==test.TaxableIncome) ##65.55


temp = pd.Series(model.predict(train[predictors])).reset_index(drop=True)

np.mean(pd.Series(train.TaxableIncome).reset_index(drop=True) == pd.Series(model.predict(train[predictors])))

