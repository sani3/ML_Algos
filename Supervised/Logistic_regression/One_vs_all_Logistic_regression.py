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


#Sigmoid (logistic) function
def sigmoid(z):
    return 1/(1+np.exp(-z))


#Get data
pathx = 'C:/Users/Sani/PYTHONPROJECTS/MachineLearning/Regression/Logistic_regression/X.csv'
pathy = 'C:/Users/Sani/PYTHONPROJECTS/MachineLearning/Regression/Logistic_regression/y.csv'
x = getdata(pathx)
y = getdata(pathy)

#Preprocess features
# x = preprocessing.scale(x)      #preprocessing.StandardScaler(x)
# x = preprocessing.add_dummy_feature(x)
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=0, shuffle=True)

#Train model
regr = linear_model.LogisticRegression(fit_intercept=True)
tetas = np.zeros((10, 401))
category = np.array(range(0,10))
for level in category:
    regr.fit(x_train, np.ravel(y_train==level))
    tetas[level, :] = np.concatenate((regr.intercept_.reshape((1,1)), regr.coef_),1)




x_test_b = preprocessing.add_dummy_feature(x_test)
z = tetas.dot(x_test_b.transpose())
prediction = sigmoid(z) >= 0.5
prediction = prediction.T
p = np.where(prediction==True, 1, 0)
d = np.zeros(y_test.shape[0])
for i in range(p.shape[0]):
    for j in range(10):
        if p[i,j]==1:
            d[i] = j
        
        

q = d == y_test
print(q.sum()/q.shape[0])

# metrics.accuracy_score(regr.predict(x_test), y_test)
# regr.score(x_test, y_test)