# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 11:57:22 2021

@author: Sani
"""
#SVM is for complex nonlinear classifiers 
#SVM uses features based on similarity defined by some kernel
#such as the linear kernel or gaussian kernel among others
#use if n_featuter is small compared to n_samples othrwise consider LR or NN
#Note: the optimization objective for SVM is always convex
from sklearn import svm, pipeline, preprocessing, datasets
x, y = datasets.make_classification(n_features=4, random_state=0)
classifier = pipeline.make_pipeline(preprocessing.StandardScaler(), svm.LinearSVC(random_state=0, tol=1e-05))
model = classifier.fit(x, y)
# svm.LinearSVC?
     # penalty='l2',
     # loss='squared_hinge',
     # *,
     # dual=True,
     # tol=0.0001,
     # C=1.0,
     # multi_class='ovr',
     # fit_intercept=True,
     # intercept_scaling=1,
     # class_weight=None,
     # verbose=0,
     # random_state=None,
     # max_iter=1000,