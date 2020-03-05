# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:43:02 2020

@author: BelluPr
"""

import numpy as np
import pandas as pd

Data=pd.read_csv("train.csv")

Data1=pd.read_csv("test.csv")

D_train_X=Data.iloc[:,:-1].values
D_train_Y=Data.iloc[:,-1:].values

test1=Data1.iloc[:,:].values




#Linear Reg
from sklearn.linear_model import LinearRegression

LR=LinearRegression()

LR.fit(D_train_X,D_train_Y)

s=LR.predict(test1)

s=np.round(s)
np.savetxt("res1.csv", s)

#log regress


from sklearn.linear_model import LogisticRegression

LR1=LogisticRegression()

LR1.fit(D_train_X,D_train_Y)

s=LR1.predict(test1)

s=np.round(s)
np.savetxt("res2.csv", s)

#dec tree

from sklearn.tree import DecisionTreeClassifier

LR2=DecisionTreeClassifier()
LR2.fit(D_train_X,D_train_Y)

s=LR2.predict(test1)

#s=np.round(s)
np.savetxt("res3.csv", s)






from sklearn.tree import DecisionTreeRegressor

LR3=DecisionTreeRegressor()
LR3.fit(D_train_X,D_train_Y)

s=LR3.predict(test1)

#s=np.round(s)
np.savetxt("res4.csv", s)



from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier(n_estimators=500,random_state=0)
RF.fit(D_train_X,D_train_Y)

s=RF.predict(test1)

#s=np.round(s)
np.savetxt("res451.csv", s)