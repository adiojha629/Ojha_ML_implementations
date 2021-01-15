# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:50:59 2021

@author: Aditya Ojha
"""

import torch
#generate tensors
x = torch.Tensor([2,4])
y = torch.Tensor([3,7])
print(x*y)
#generate arrays of zeros
x = torch.zeros([2,4])
print(x)
#generate arrays of random numbers
y = torch.rand([4,5])
print(y)
#flatten arrays or reshape
y = y.view([1,20])
print(y)
