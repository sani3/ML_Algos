# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:02:31 2021

@author: Sani
"""
# Dimensionality Reduction finds a lower dimensional surface
# on which to project a higher dimensional data onto
# such that the projection error is minimized
# Dimensionality reduction is usefull for data compression to 
# enable visualization of high dimensional data and to speed up learning algorithm

# Principal Component Analysis is a dimensionality reduction algorithm 
# Generally, choose k, number of pc to reduce n features to, such that 
# 99% of variance is retained which can be computed from svd of features

# To do pca, start with mean normalization/feature scaling


import numpy as np

def normalize_features(data):
    mu = data.mean(axis=0)
    sigma = data.std(axis=0)
    data_norm = np.divide(data-mu, sigma)
    return data_norm, mu, sigma

def covariance_matrix(data_norm):
    cov_mat =  np.dot(data_norm.transpose(), data_norm) / data_norm.shape[0]
    return cov_mat

def si_va_de(cov_mat):
    u, s, v = np.linalg.svd(cov_mat)
    return u, s, v

def reduce(data_norm, u, k):
    u = u[:, 0:k]
    red = np.dot(data_norm, u)
    return red
    
    
#######Dimensionality Reduction with PCA########
# import pandas as pd
# data = pd.read_clipboard() #add headers
# data = data.to_numpy()
# data_norm, mu, sigma = normalize_features(data)
# cov_mat = covariance_matrix(data_norm)
# u, s, v = si_va_de(cov_mat)
# red = reduce(data_norm, u, 1)
################################################