# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:28:04 2020
Least squares regression 

@author: adioj
"""

import pandas as pd #for data stripping
import numpy as np #for math operations
import matplotlib.pyplot as plt #For graphing the data

data = pd.read_csv('data.csv',header = None) #Import the data
X = data.iloc[:, 0] #Variable X gets all the x values (values in first column)
Y = data.iloc[:, 1] #Variable Y gets all y values (values in second column)


""" Building the model
Least Squares regression actually has a formula; so not backpropagation needed
"""
X_mean = np.mean(X)
Y_mean = np.mean(Y)
m_numerator = 0
m_denominator = 0

for x,y in zip(X,Y):
    m_numerator += (x-X_mean)*(y-Y_mean)
    m_denominator += (x-X_mean)*(x-X_mean)
    
m = m_numerator / m_denominator #Slope
c = Y_mean - m*X_mean #Y-intercept


star_string = "\n************\n"
print(star_string+"Least Squares Regression Line" +star_string)
print(star_string+"Slope is " + str(m) + ", Y-intercept is " + str(c)+star_string)

Y_pred = m*X + c

"""Show Data on Graph"""

plt.scatter(X,Y)
plt.plot(X,Y_pred,color = 'red')
plt.show()