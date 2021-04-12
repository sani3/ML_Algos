# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:56:43 2020

Implementation of normal equation for linear regression

@author: Sani
"""

import numpy as np

#Get data
def getdata(path):
    #read data from text file into numpy array
    data = np.genfromtxt(path, delimiter = ",")
    return data


#get features
def getfeatures(data):
    n_samples, n_features = data.shape
    #get array of uotcomes from data
    y = data[:, -1].copy().reshape((n_samples,1))
    #get array of features from data
    x = np.delete(data, -1, 1)
    return x, y


#add bias feature
def bias_features(x):
    n_samples, n_features = x.shape
    #concatenate the intercept term, a column of ones, with x_norm
    x = np.concatenate((np.ones((n_samples, 1)), x), axis=1)
    return x

def normal_equation(x, y, lmd=0.01):
    n = x.shape[1]
    d = np.diag(np.ones(n))
    d[0,0] = 0
    tetas = np.linalg.pinv(np.add(x.transpose().dot(x), np.multiply(lmd, d))).dot(x.transpose().dot(y))
    return tetas



path = 'ex1data2.txt'
data = getdata(path)
x, y = getfeatures(data)
x = bias_features(x)
tetas = normal_equation(x, y)
print("tetas: ", tetas)

x_new = np.array([1650,3]).reshape((1,2))
x_val = np.concatenate((np.ones((x_new.shape[0],1)), x_new), axis=1)
prediction = x_val.dot(tetas)
print("prediction: ", prediction)