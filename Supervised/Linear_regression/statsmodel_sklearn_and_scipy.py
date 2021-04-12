# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 06:23:17 2021

@author: Sani
"""

import numpy as np
from sklearn.model_selection import train_test_split


def getdata(path):
    #read data from text file into numpy array
    data = np.genfromtxt(path, delimiter = ",")
    return data


#get features
def getfeatures(data):
    m,n = data.shape
    #get array of uotcomes from data
    y = data[:, -1].copy().reshape((m,1))
    #get array of features from data
    x = np.delete(data, -1, 1)
    #initialize tetas
    tetas = np.zeros((n,1))
    return x, tetas, y

#normalize features
def normalize_features(x):
    m = x.shape[0]
    #normalize features
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    xnorm = np.divide(x-mu, sigma)
    #concatenate the intercept term, a column of ones, with x_norm
    x_norm = np.concatenate((np.ones((m,1)), xnorm), axis=1)
    return x_norm, mu, sigma


path = 'ex1data2.txt'
data = getdata(path)
x, tetas, y = getfeatures(data)
x_norm, mu, sigma = normalize_features(x)





##Statsmodel
import statsmodels.api as sm
model = sm.OLS(y, x_norm).fit()
print(model.summary())

#predict 1650, 3
pred = model.predict(np.concatenate(([1], np.array([1650, 3]-mu)/sigma),0))
print("Predicting [1650, 3] as : ", pred)

##sklearn
from sklearn import linear_model
regr = linear_model.LinearRegression(fit_intercept=True, normalize=True)
regr.fit(np.concatenate((np.ones((x.shape[0],1)), x), axis=1), y)
regr.intercept_
regr.coef_
regr.score(np.concatenate((np.ones((x.shape[0],1)), x), axis=1), y)
regr.predict(np.array([1, 1650, 3]).reshape(1,-1))


##other options
#scipy optimize (minimize, curve_fit, etc)
