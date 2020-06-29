#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 20:52:42 2020

@author: albwrekt

This is a Self Organizing Map for the Udemy Class Deep Learning by Super Data Science.

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

# Columns are attributes
# Rows are customers
# This produces dimensional 2D map
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# training the model only uses the x set because it's unsupervised
# feature scaling needed as well
sc = MinMaxScaler(feature_range=(0,1))

# fit the objects to the X and apply normalization
x = sc.fit_transform(x)

# Training the som - This will be done with MiniSOM
# size of x is used here. The id is needed to id people
# learning rate - higher is faster convergence
# decay_function can be used to help convergence
# sigma is the distance of the neighborhood
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(data=x)

# num_iteratiosn will be the times the data is iterated
som.train_random(data=x, num_iteration=100)

# visualizing the results
# winning nodes will have specific colors
bone()

# highest intermediate distance between nodes are white (most distance)
# lowest intermediate distance between nodes are dark
# the highest ones are the outliers because they are not part of immediate clusters
pcolor(som.distance_map().T)
colorbar()

# adding markers
# red circle -  didn't get approval
# green square - did get approval
markers = ['o', 's']
colors = ['r', 'g']

# loop through and identify customers approval status
# i - index of customers
# x_i - vector of customers
for i, x_i in enumerate(x):
    w = som.winner(x_i)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2
         )
show()

# finding the frauds through the winning nodes map
mappings = som.win_map(x)

# find the coordinates that correspond to the fraud outliers
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis=0)
frauds = sc.inverse_transform(frauds)
print(frauds)

    