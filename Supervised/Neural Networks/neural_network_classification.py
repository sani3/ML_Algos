# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:56:43 2020

Implementation of batch gradient descent for linear regression

@author: Sani
"""

import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics

#Get data
def getdata(path):
    #read data from text file into numpy array
    data = np.genfromtxt(path, delimiter = ",")
    return data


#one-hot encode
def one_hot(y):
    l = np.unique(y)
    yoh = np.zeros((y.shape[0], l.shape[0]))
    for i in range(yoh.shape[0]):
        for j in range(yoh.shape[1]):
            if j == y[i]:
                yoh[i, j] = 1
    return yoh


#Sigmoid (logistic) function
# z = x_norm.dot(tetas.transpose)
def sigmoid(z):
    return 1/(1+np.exp(-z))


#sigmoid gradient
def sigrad(z):
    return sigmoid(z)*1-sigmoid(z)


def init_tetas(m, n):
    epsilon=0.12
    return np.random.random((m, n))*2*epsilon-epsilon


#Get data
pathx = 'C:/Users/Sani/PYTHONPROJECTS/MachineLearning/Regression/Logistic_regression/X.csv'
pathy = 'C:/Users/Sani/PYTHONPROJECTS/MachineLearning/Regression/Logistic_regression/y.csv'
x = getdata(pathx)
y = getdata(pathy)


#Preprocess features
# x = preprocessing.scale(x)      #preprocessing.StandardScaler(x)
# x = preprocessing.add_dummy_feature(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0, shuffle=True)
#one-hot encode y
y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)


#Network architecture
#Using a three layer network with 25 hidden layers and 10 output layers
#with 400 input units less bias unit,
layers = 3
n_samples, n_features = x_train.shape
units1 = n_features
units2 = 25
units3 = 10
tetas1 = init_tetas(units2, units1+1) #(25, 401)
tetas2 = init_tetas(units3, units2+1) #(10, 26)


#forward propagaion
a1 = preprocessing.add_dummy_feature(x_train) #(3500, 401)
z1 = a1.dot(tetas1.T)
a2 = preprocessing.add_dummy_feature(sigmoid(z1)) #(3500, 26)
z3 = a2.dot(tetas2.T)
a3 = sigmoid(z3) #(3500, 10)


#compute cost
lmd = 1
n_samples, n_features = a3.shape
cost = (-1/n_samples) * ((y_train_oh * np.log(a3)) + ((1-y_train_oh) * (1-np.log(a3)))).sum() + (lmd/(2*n_samples))*np.add((np.square(tetas2.T[1:,:])).sum(), (np.square(tetas1.T[1:,:])).sum())
    

#back propagation
deltas3 = a3-y_train_oh ##(3500, 10)
deltas2 = deltas3.dot(tetas2) * sigrad(a2) ##(3500, 26)
# deltas1 = deltas2.dot(tetas1) * sigrad(a1) ##(3500, 401)


tetas1_grad = np.zeros_like(tetas1) #(25, 401)
tetas2_grad = np.zeros_like(tetas2) #(10, 26)

n_samples, n_features = a3.shape

Jder2 = deltas3.T.dot(a2) #(10, 26)
tetas2_grad[:, 0] = (1/n_samples) * Jder2[:, 0]
tetas2_grad[:,1:] = (1/n_samples) * Jder2[:,1:] + (lmd/n_samples) * tetas2[:,1:]

Jder1 = deltas2[:,1:].T.dot(a1) #(25, 401)
tetas1_grad[:,0] = (1/n_samples) * Jder1[:, 0]
tetas1_grad[:,1:] = (1/n_samples) * Jder1[:, 1:] + (lmd/n_samples) * tetas1[:,1:]


#### Stucked with putting it all together