# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:56:43 2020

Implementation of batch gradient descent for linear regression
with normalization.

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
    #initialize tetas
    tetas = np.zeros((n_features))
    return x, tetas, y


#normalize features
def normalize_features(x):
    #normalize features
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    xnorm = np.divide(x-mu, sigma)
    return xnorm, mu, sigma


#add bias feature
def bias_features(xnorm):
    n_samples, n_features = xnorm.shape
    #concatenate the intercept term, a column of ones, with x_norm
    x_norm = np.concatenate((np.ones((n_samples, 1)), xnorm), axis=1)
    return x_norm


#Linear function
def linear(z):
    return z


# compute cost
def costfunc(tetas, x_norm, y, lmd=1):
    n_samples, n_features = x_norm.shape
    z = tetas.dot(x_norm.transpose())
    h = linear(z)
    e = np.subtract(h, y.T)
    se = np.square(e)
    cost = np.multiply(1/n_samples, np.add(se.sum(), (lmd/2)*np.square(tetas[1:]).sum()))
    return cost


#compute gradient
def gradfunc(tetas, x_norm, y, lmd=1):
    n_samples, n_features = x_norm.shape
    z = tetas.dot(x_norm.transpose())
    h = linear(z)
    e = np.subtract(h, y.T)
    grad = np.zeros_like(tetas)
    grad[0] = np.multiply(1/n_samples, np.dot(e, x_norm[:,0]))
    grad[1:] = np.multiply(1/n_samples, np.add(np.dot(e, x_norm[:,1:]), lmd*tetas[1:]))
    return grad


#learn tetas
def grad_descent(tetas, x_norm, y, iterations = 99000, alpha = 0.01):
    
    costs = np.zeros((iterations, 2))
    
    #Batch Gradient Descent
    for i in range(iterations):
        cost = costfunc(tetas, x_norm, y)
        costs[i, 0] = i
        costs[i, 1] = cost
        tetas = np.subtract(tetas, np.multiply(alpha, gradfunc(tetas, x_norm, y)))
    return tetas, costs


##################################################################
#Get data
path = 'ex1data2.txt'
data = getdata(path)
x, tetas, y = getfeatures(data)
xnorm, mu, sigma = normalize_features(x)
x_norm = bias_features(xnorm)

#learn tetas
tetas, costs = grad_descent(tetas, x_norm, y)
print("tetas: ", tetas)

#predict n_new
x_new = np.array([1650,3]).reshape((1,2))
x_val = np.concatenate((np.ones((x_new.shape[0],1)), (x_new-mu)/sigma), axis=1)
z = tetas.dot(x_val.transpose())
prediction = linear(z)
print("prediction: ", prediction)


# NOTES: linear regression
# *Consider creating a new feature ie, using Area instead of length and breadth
# *Consider polynomial function that could fit the curve better eg, quadratic, cubic, quadratic and square root term, etc
# *Beware of overfitting and underfitting