#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:13:41 2018

@author: pmehta
"""

import numpy as np
import pandas as pd
import os
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath)

import case
import anova
import pod
import krigging




cfd_data, U1, x1, y1 = case.define_case.case1()

f_anova = anova.cANOVA.decomposition(cfd_data, anchor = [2,2,2])


shape = f_anova.shape

S1, S2, S3 = shape[0], shape[1], shape[2]

f_data_matrix = np.zeros((S1*S2,S3))

for k in range(S3):
    a = 0
    for i in range(S1):
        for j in range(S2):
            f_data_matrix[a,k] = cfd_data[i,j,k]
            a += 1



f_pod, coeff_v, Phi = pod.POD.pod(f_data_matrix, 0.98)


#Krigging

coef_v1 = coeff_v[:,0]
coef_v2 = coeff_v[:,1]

t = np.linspace(0,5,6,endpoint = True)

discretisation, mu_1, var_1 = krigging.gaussion_process_regression.fit(t,coef_v1,Kernel = "poly_cubic_spline")

discretisation, mu_2, var_2 = krigging.gaussion_process_regression.fit(t,coef_v2,Kernel = "poly_cubic_spline") 





#----------------------------------------------------------------------
## TEsting C_APK
        
i = j = k = 0




f_raw_pca_test = np.zeros((S1,S2))

#u_test = np.zeros((R*R)) #delete or modify. Line copied form previous verison

for i in range(S1):
    for j in range(S2):
        f_raw_pca_test[i,j] = f_pod[k,3]
#        u_test[k] = f_raw_pca_test[i,j]
#        error =  u_test[k] - u1[k]
#        print(error, k)
        k = k+1

M,N = f_raw_pca_test.shape

i = j = k = 0


# post processing

#x1 = np.linspace(0, 20, 21, endpoint=True)

#y1 = x1



"""
X and Y must both be 2-D with the same shape as Z, 
or they must both be 1-D such that len(X) is the number of columns in Z and len(Y) is the number of rows in Z.

Use masking for non rectangilar girds

"""

plot1 = plt.contour(x1,y1,f_raw_pca_test)

plot2 = plt.contourf(x1,y1,f_raw_pca_test)

R1 = mu_1.shape

R1 = R1[0]

R2 = 441


f_kriggig_test = np.zeros((R2,R1))

for i in range(R2):
    for j in range(R1):
        f_kriggig_test[i,j] = (mu_1[j]*Phi[i,0]) + (mu_2[j]*Phi[i,1])


f_raw_pca_test = np.zeros((S1,S2))

i = j = k = 0
for i in range(S1):
    for j in range(S2):
        f_raw_pca_test[i,j] = f_kriggig_test[k,3]
        k += 1



plot3 = plt.contourf(x1,y1,f_raw_pca_test)