# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:50:27 2020

implementation of adam

@author: adioj
"""
import numpy as np
num_iterations = 100
w = 10
beta_1 = .9
beta_2 = .999
m = 0
v = 0
epsilon = .2
step_size = 1
def compute_gradient(x,y):
    return x-y/(y^2)

for t in range(num_iterations):
    x = 13
    y = 10
    g = compute_gradient(x,y)
    m = beta_1 * m + (1-beta_1) * g
    v = beta_2 * v + (1-beta_2) * np.power(g,2)
    m_hat = m / (1 - np.power(beta_1, t))
    v_hat = v / (1 - np.power(beta_2, t))
    w = w - step_size *m_hat / (np.sqrt(v_hat) + epsilon)
    