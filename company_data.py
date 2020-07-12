# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:56:42 2020

@author: Varun
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder as le

data=pd.read_csv("F:\\EXCEL R\\ASSIGNMENTS\\decision tress\\Company_Data.csv")
###function to convert
###def datatypes(df):
##     a= df.dtypes
 #    b=dict(a)
##     le=LabelEncoder()
##     for key in b.keys():
##        if b[key]==np.object:
#            df[key]=le.fit_transform(df[key])
#    return df
data.Sales=pd.cut(data.Sales,bins=[0,4,8,12],labels=['A','B','C'])
data=data.dropna(subset=['Sales'])
le=LabelEncoder()

data['ShelveLoc']=le.fit_transform(data['ShelveLoc'])
data['Urban']=le.fit_transform(data['Urban'])
data['US']=le.fit_transform(data['US'])

colnames=list(data.columns)
predictors=colnames[1:]
target=colnames[0]

train,test=train_test_split(data,test_size=0.3,random_state=0)
model= DecisionTreeClassifier(criterion="entropy")
model.fit(train[predictors],train[target])


preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)
np.mean(preds==test.Sales) # Accuracy = Test

temp = pd.Series(model.predict(train[predictors])).reset_index(drop=True)

np.mean(pd.Series(train.Sales).reset_index(drop=True) == pd.Series(model.predict(train[predictors])))
