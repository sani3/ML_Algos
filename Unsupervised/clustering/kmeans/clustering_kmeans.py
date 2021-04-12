# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:01:47 2021

@author: Sani
"""

# kmean clustering agorithm iteratively assigns samples to one of k clusters
# such that a sample is closer to the cluster it is assigned to than the remaining clusters
# For all the samples in each cluster, a centoid is computed which becomes
# the new position of that cluster and the process is repeated over and over again

import numpy as np


def init_centroids(data, k):
    ind = np.random.choice(data.shape[0],k,False)
    centroids = data[ind]
    return centroids


def mag_func(data):
    mag = np.sqrt(np.square(data).sum(1))
    return mag


def cluster(data, centroids):
    stub = np.zeros((data.shape[0], centroids.shape[0]))
    for i in range(centroids.shape[0]):
        diffs = data - centroids[i,:]
        mag = mag_func(diffs)
        stub[:, i] = mag
    min_mag = stub.min(1)
    for i in range(stub.shape[0]):
        for j in range(stub.shape[1]):
            if stub[i, j] == min_mag[i]:
                stub[i, j] = j
            else:
                stub[i, j] = 0
    return stub.sum(1)


def move_centroid(data, stub, centroids):
    u = np.unique(stub)
    for i in range(centroids.shape[0]):
        d = data[stub==u[i]]
        centroids[i] = d.mean(0)
    return centroids



def kmn(data, k, it):
    centroids = init_centroids(data, k)
    for i in range(it):
        stub = cluster(data, centroids)
        centroids = move_centroid(data, stub, centroids)
    return stub


########kmeans clustering action########
# import pandas as pd
# k = 3
# it = 10
# data = pd.read_clipboard() #add headers
# data = data.to_numpy()
# stub = kmn(data, k, it)
# data = np.concatenate((data, stub), 1)
########################################