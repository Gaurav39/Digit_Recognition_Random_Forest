# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:44:44 2018

@author: Gaurav
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as pt

data = pd.read_csv('E:/kaggle/digitrecognition/dataset/train.csv')
X_test = pd.read_csv('E:/kaggle/digitrecognition/dataset/test.csv')
#data.head() 
#a = data.iloc[4,1:].values
#a = a.reshape(28,28).astype('uint8')
#pt.imshow(a)
X = data.iloc[:,1:]
Y = data.iloc[:,0]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)

pred = rf.predict(x_test)
s = y_test.values
count = 0

for i in range(len(pred)):
    if (pred[i]==s[i]):
        count = count + 1
print (count)
print (len(pred))

y_pred = rf.predict(X_test)

