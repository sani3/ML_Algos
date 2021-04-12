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
from sklearn.neural_network import MLPClassifier

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
y_oh = one_hot(y)
y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)


#Network architecture
#Using a three layer network with 25 hidden layers and 10 output layers
#with 400 input units less bias unit,
clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(25,), random_state=1)
clf.fit(x_train, y_train_oh)
p = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test_oh, p)
print("accuracy: ", accuracy)
