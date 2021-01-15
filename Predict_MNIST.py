# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:56:04 2021
Building Neural Network with pytorch
@author: Aditya Ojha
"""
#Import Libraries
## For importing data
import torch
#import torchvision
from torchvision import transforms, datasets
## For building network
import torch.nn as nn #class
import torch.nn.functional as F #set of functions
## For backpropagation
import torch.optim as optim
## For visualization
import matplotlib.pyplot as plt

#Get raw, image data
train = datasets.MNIST("",train=True,transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("",train=False,transform=transforms.Compose([transforms.ToTensor()]))
#Clean data into Tensors, batch sizes, and shuffle
trainset = torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset = torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #define fully connected (fc) layers
        self.fc1 = nn.Linear(28*28 #size of images
                             , out_features = 64 #arbitrary
                             )
        self.fc2 = nn.Linear(64, out_features=64)
        self.fc3 = nn.Linear(64,64) #64's are arbitrary
        self.fc4 = nn.Linear(64,10) #10 classes
    #Feed Forward Method
    def forward(self,x):
        x=F.relu(self.fc1(x)) #pass through first layer, then activation
        x=F.relu(self.fc2(x)) #pass through second layer
        x=F.relu(self.fc3(x)) #pass through third layer
        x=self.fc4(x) #pass through fourth layer
        #use softmax so we get probability distribution
        #dim =1 , remember that data is list of input tensor, output (label) tensor.
        #We want to get the probability distribution over the classes (index 1 of list)s
        return F.log_softmax(x, dim = 1) #return output
net = Net()
print(net)
#generate random image
X = torch.rand((28,28)).view(-1,28*28) #-1 means that the input batch size could be any length
print(net(X))
optimizer = optim.Adam(net.parameters() #gets all adjustable parameters in net
                       ,lr = 0.001)

#training time:
EPOCHS = 3 #3 passes through all of our data
for epoch in range(EPOCHS):
    for data in trainset: 
        #data is batches of ten
        X, y = data
        net.zero_grad() #make gradient zero (clears gradient calculated from last batch)
        output = net(X.view(-1,28*28))
        loss = F.nll_loss(output,y) #b/c data is a value (ie. 4) will look into later
        loss.backward() #back propagate 
        optimizer.step() #apply gradient to each parameter
    print(loss)

#how did we do in training?
correct = 0
total = 0
with torch.no_grad(): #don't calculate gradients
    for data in trainset:
        X, y = data
        output = net(X.view(-1,28*28))
        for idx,i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct+=1
            total+=1
print("Accuracy: ", round(correct/total,3))
#let's see an example
plt.imshow(X[0].view([28,28]))
plt.show()
prediction = torch.argmax(net(X[0].view(-1,28*28)))
print("Prediction is",prediction)