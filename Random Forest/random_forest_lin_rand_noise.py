# -*- coding: utf-8 -*-
"""
Created on Sun May  3 12:16:18 2020
Use of random forest on a set of linear data
Data Description:
    Y = mX + b + random_noise
@author: adioj
"""
"""Libraries to import"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

"""Import Data"""
data = pd.read_csv("classifer_data.csv", header = None)
x_raw = data.iloc[:,0]
y_raw = data.iloc[:,1]

ones = []
zeros = []
for x,y in zip(x_raw,y_raw):
    if(y == 1):
        ones.append(x)
    else:
        zeros.append(x)
X = [ones,zeros]
Y = y_raw       

model = RandomForestClassifier(n_estimators = 100)
model.fit(X,Y)
Y_predit = model.predict(X)
