# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:03:20 2021

@author: Aditya Ojha
"""

#Importing data using torchvision
import torch
import torchvision
from torchvision import transforms, datasets
#collection of datasets for vision

train = datasets.MNIST("",
                       train = True,
                       download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("",
                       train = False,
                       download=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
#Make data into tensors: use utils.data.DataLoader
trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset = torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)
#batch_size= num of samples passed into model before weight updates

#iterate through data
for data in trainset:
    print(data)
    break
#'data' is a group of ten images (of handwritten digits) and then the correct labels for the images
x,y = data[0][0],data[1][0] #get image (x) and label (y)
print("Image of a ",y)
import matplotlib.pyplot as plt
#x is a 1,28,28
print(x.shape)
#to print image it needs to be 28 by 28; use view() to reshape it
x = x.view([28,28])
plt.imshow(x)

#How to balance data

#First check relative frequencies of each class

## initialize counter
possible_lables = [1,2,3,4,5,6,7,8,9,0]
count_dict = dict()
for label in possible_lables:
    count_dict[label] = 0

## count
for data in trainset:
    _,labels = data
    for label in labels:
        count_dict[int(label)]+=1 #increment counter
print(count_dict)