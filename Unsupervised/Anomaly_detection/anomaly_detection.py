# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 21:54:17 2021

@author: Sani
"""

# Anomaly detection uses the gaussian distribution such that 
# model is product of p(x; mu, sigma) <= epsilon for each feature
# use if positive (anomalous) sample is very small compared to negative (normal samples)
# also use if there are many possible anomalies such that current data did not capture all
# consider transformations to ensure gaussian distribution of each feature
# See multivariate gaussian distribution p(x; mu, np.diagflat(sigma))   ###??np.linalg.svd(sigma)[2]
import numpy as np

# get data
# ensure gaussian features, use transformation eg log(x+c), x**c, etc

def anomaly_params(data):
    #data is an array of features
    mu = np.mean(data, 0)
    sigma = np.sdt(data, 0)
    return mu, sigma

def density(x, mu, sigma):
   prob = np.prod((1/(np.sqrt(2*np.pi*sigma)))*np.exp(-np.square(x-mu)/(2*np.square(sigma))), 1)
   return prob

def detect_anomaly(prob, epsilon):
    return prob < epsilon



####################################################
#For multivariate gaussian distribution
####################################################
# def anomaly_params(data):
#     #data is an array of features
#     mu = np.mean(data, 0)
#     sigma = np.sdt(data, 0)
#     sigma = np.diagflat(sigma)
#     return mu, sigma

# def density(x, mu, sigma):
#     n_samples, n_features = x.shape
#     prob = (1/(2*np.pi)**(n_features/2)*np.sqrt(np.det(sigma)))*np.exp(-(1/2)*np.sum((x-mu).T.dot(np.linalg.pinv(sigma))*(x-mu)))
#     return prob

# def detect_anomaly(prob, epsilon):
#     return prob < epsilon