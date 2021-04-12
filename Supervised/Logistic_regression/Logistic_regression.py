# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 16:56:43 2020

Implementation of batch gradient descent for linear regression

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



#Sigmoid (logistic) function
def sigmoid(z):
    return 1/(1+np.exp(-z))


# compute cost
def costfunc(tetas, x_norm, y, lmd=1):
    n_samples, n_features = x_norm.shape
    z = tetas.dot(x_norm.transpose())
    h = sigmoid(z)
    cost = (-1/n_samples) * ((y.T * np.log(h)) + ((1-y.T) * (1-np.log(h)))).sum() + (lmd/(2*n_samples))*(np.square(tetas[1:])).sum()
    return cost


#compute gradient
def gradfunc(tetas, x_norm, y, lmd=1):
    n_samples, n_features = x_norm.shape
    z = tetas.dot(x_norm.transpose())
    h = sigmoid(z)
    e = np.subtract(h, y.T)
    grad = np.zeros_like(tetas)
    grad[0] = np.multiply(1/n_samples, np.dot(e, x_norm[:,0]))
    grad[1:] = np.multiply(1/n_samples, np.add(np.dot(e, x_norm[:,1:]), lmd*tetas[1:]))
    return grad


#learn tetas
def grad_descent(tetas, x_norm, y, iterations = 100000, alpha = 0.01):
    
    costs = np.zeros((iterations, 2))
    
    #Batch Gradient Descent
    for i in range(iterations):
        cost = costfunc(tetas, x_norm, y)
        costs[i, 0] = i
        costs[i, 1] = cost
        tetas = np.subtract(tetas, np.multiply(alpha, gradfunc(tetas, x_norm, y)))
    return tetas, costs





#Optimization algorithms include
#Gradient descent: You implemet cost, derivative of cost and choose alpha
#Conjugate gradient: You only provide implemetation for cost, derivative of cost
#BFGS: You only provide implemetation for cost, derivative of cost
#L-BFGS: You only provide implemetation for cost, derivative of cost

#For multiclass classification, consider One-vs-all

#   Underfitting: Hypothesis is a loose fit to data (High bias)
#   Overfitting: Hypothesis is too rigidly fit to data (High variance)
#       Consider regularization to address overfitting









##################################################################
#Get data
path = 'ex2data2_polymap_features.txt'
path = 'ex2data1.txt'
data = getdata(path)
x, tetas, y = getfeatures(data)
xnorm, mu, sigma = normalize_features(x)
x_norm = bias_features(xnorm)

#learn tetas
tetas, costs = grad_descent(tetas, x_norm, y)
print(tetas)


#predict n_new
#x_new = np.array([45, 85]).reshape((1,2)) #
x_new = np.array(x)
x_val = np.concatenate((np.ones((x_new.shape[0],1)), (x_new-mu)/sigma), axis=1)
zp = tetas.dot(x_val.transpose())
prediction = sigmoid(zp)
prediction = prediction >= 0.5
print("train_accuracy: ", np.mean(prediction==y))



