#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################
# File: linear_regresion.py   #
#                             #
# Author: Marta Urbaniak      #
#                             #
# Date: 4.05.2019             #
#                             #
# Description:                #
#    Normal equation method   #
#    to find best linear      #
#    regresion parameters     #
################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv

data = np.loadtxt('data/ex1data2.txt', delimiter=',')
X = np.c_[data[:,0], data[:, 1]]
y = np.c_[data[:,2]]
m = y.size

X = np.c_[np.ones(m), X]

theta = np.dot(np.dot(inv(np.dot(np.transpose(X),X)),np.transpose(X)),y)

print("theta: ",theta.ravel())
