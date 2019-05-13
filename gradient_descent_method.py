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
#    Gradient descent method  #
#    to find best linear      #
#    regresion parameters     #
###############################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('data/data.txt', delimiter=',')

X = np.c_[data[:,0], data[:, 1]]
y = np.c_[data[:,2]]
m = y.size 

alpha = 0.01

num_iters = 400

theta = np.zeros((X.shape[1], 1))

#normalization

mean_1 = 0
mean_2 = 0
st_dev_1 = 0
st_dev_2 = 0

mean_1 = np.mean(data[:,0])
mean_2 = np.mean(data[:,1])
st_dev_1 = np.std(data[:,0])
st_dev_2 = np.mean(data[:,1])

                  
X = np.c_[(data[:,0]-mean_1)/st_dev_1, (data[:, 1]-mean_2)/st_dev_2]

X = np.c_[np.ones(m), X]

def linearFunction(X, theta):
   
    f = 0
    
    f = np.dot(X,theta)
    
    return f
  
  
def computeCostMult(X, y, theta):
    
    J = 0
    m = y.size
    f = 0
    
    h = linearFunction(X,theta)
    
    
    J = ((sum((h-y)**2))/(2*m))
    
    return(J)


def gradientDescent(X, y, theta, alpha, num_iters):
    
    m = y.size
    theta = np.zeros((X.shape[1], 1))
    J_history = np.zeros(num_iters)
    
    X_T = np.transpose(X)
    
    h = linearFunction(X,theta)
    
    for iter in np.arange(num_iters):
        
        a = theta - (alpha/m)*(np.dot(X_T,(h-y)))

        theta = a
        
        
        h = linearFunction(X,a)
        
        J_history[iter] = computeCostMult(X, y, theta)
    
    return(theta, J_history)
  
  
theta , Cost_J = gradientDescent(X, y, theta, alpha, num_iters)
print('theta: ',theta.ravel())

plt.plot(Cost_J)
plt.ylabel('Koszt J')
plt.xlabel('Iteracje');

plt.savefig("data_visualisation/wykres_koszt(iteracji).png")
 