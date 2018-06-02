#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:15:24 2018

@author: pmehta
"""

import numpy as np
import scipy as sp
from scipy import linalg

"""
    This file preform POD in N Dimensions

"""

class POD:
    
    def pod(f_data_matrix, pod_treshold):
        
        shape = f_data_matrix.shape
        
        M, N = shape[0], shape[1]
        
        Snapshot_matrix = np.dot(f_data_matrix, (np.eye(N) - np.dot(np.ones(N), 1/N)))
        
        
        #SVD Decompisition 
        
        U,sig,Vt = linalg.svd(Snapshot_matrix)
        
        
        #Private Variables
        explained_var = trunc_order = sumer = 0
        
        
        

        while (explained_var <= pod_treshold):
            sumer += sig[trunc_order]
            explained_var =  sumer/sig.sum()
            trunc_order += 1


        r = trunc_order - 1 #looping effects - code needs imporvement
        

        # Performing Truncation - SVD


        if (r == 0):
            sig_r = np.zeros((1))
        
            sig_r[r] = sig[r]
        
        else:
    
            sig_r = np.zeros((trunc_order,trunc_order))
        
    
    
        for i in range(trunc_order):
            sig_r[i,i] = sig[i]


        U_r = np.zeros((M,trunc_order))
        Vt_r = np.zeros((trunc_order,N))

        U_r = U[:,:trunc_order]
        Vt_r = Vt[:trunc_order,:] #check well


        #Buliding POD basis vectors

        Phi = np.dot(U_r, sig_r)
        
        #Phi = Phi[:,0]



        #Computing coeff. for K-L Expansion
        
        #------------------------------------------------------------------------

        # Correaltion matrix

        Corr_matrix = (1/N)*np.dot(Snapshot_matrix.transpose(), Snapshot_matrix)
        
        
        #coeff_v_eig_val, coeff_v_eig_vector = np.linalg.eig(Corr_matrix)

        #Finding the right eigen vectors
        coeff_v_eig_val, coeff_v_eig_vector = sp.linalg.eig(Corr_matrix, right=True)


        #Computing Coeff.

        #coeff_v = coeff_v_eig_vector[0,:]

        #KL Expansion

        result = np.zeros((M,N))


        for i in range(M):
            for j in range(N):
                sumer = 0
                for k in range(trunc_order):
                    sumer = sumer + coeff_v_eig_vector[j,k]*Phi[i,k]
                result[i,j] = sumer
        
        
        return result, coeff_v_eig_vector, Phi
