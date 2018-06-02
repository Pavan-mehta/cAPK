#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:41:16 2018

@author: pmehta

Email: mehtapavanp@gmail.com
"""

import numpy as np

class cANOVA:
    
    """
    cANOVA class is for anchored ANOVA decomposition
    
    The data is feeded as an array: array[dimension 1, dimension 2, dimension 3]
    
    This file is written for problem is 3 dimension with truncation dimention = 2
    
    However, it can be readily exptended
    
    TO DO: Generic code for N - Dimension
    
    """
    
    def __init__(self, cfd_data, anchor):
        self.cfd_data = cfd_data
        self.anchor = anchor
    
    def compute_f0(cfd_data, anchor):
        
        
        """
        This function gets the mean or funcatinal value at anchored point.
        
        If the anchored point is not specified then it takes the center of compuational domain
        
        TO DO: Selceting center based upon input uncertain range, which is better suitaed for uniform and non - unifrom sampling techniques
        
        
        Important variables:
        
        f0: Mean or evaluated at anchored point
        
        anchor: array for anchor points
        
        n_dim: Total number of dimensions
                      
        """
        
        
        shape = cfd_data.shape
            
        n_dim = np.count_nonzero(shape)
        
        #----------------------
        #Improve this lines of code of selction of center
        
        if anchor is None:
                                    
            anchor = np.zeros((n_dim))
            
            for s in range(n_dim):
                            
                anchor[s] = shape[s]/2 #Not suitable for non - uniform sampling
        else:
            print("selecting f0 based on anchor points provided")
        
        #----------------------
        
        anc1, anc2, anc3 = anchor
        
        f0 = cfd_data[anc1, anc2, anc3] #Not suitable for non - uniform sampling

        return f0, n_dim, shape, anchor
    


    def decomposition(cfd_data, anchor):
        
        """
        This function evalutes 1st and 2nd order terms as well as perform anova decompostion
        
        Important variables:
            
            1st order terms:
                
                f1, f2, f3
            
            2nd order terms:
                
                f12, f13, f23
                
            anchor point: anc1, anc2, anc3
                       
        
        To DO: Extention in N Input Dimension and specifiying trunction dimension
        
        """
        
        
        f0, n_dim, shape, anchor = cANOVA.compute_f0(cfd_data, anchor)
                
        anc1, anc2, anc3 = anchor
        
        S1, S2, S3 = shape
        
        f1 = np.zeros(S1)
        f2 = np.zeros(S2)
        f3 = np.zeros(S3)

        f12 = np.zeros((S1,S2))
        f13 = np.zeros((S1,S3))
        f23 = np.zeros((S2,S3))
        
        f_anova = np.zeros((S1,S2,S3))
        
        for i in range(S1):
            for j in range(S2):
                for k in range(S3):
                                        
                    #--------------------------------------------------
                    #First order terms
                    #--------------------------------------------------
                                        
                    f1[i] = cfd_data[i,anc2,anc3] - f0
                    f2[j] = cfd_data[anc1,j,anc3] - f0
                    f3[k] = cfd_data[anc1,anc2,k] - f0
        
                    #--------------------------------------------------
                    #Second order terms
                    #--------------------------------------------------
                
                    f12[i,j] = cfd_data[i,j,anc3] - f1[i] - f2[j] - f0
                    f13[i,k] = cfd_data[i,anc2,k] - f1[i] - f3[k] - f0
                    f23[j,k] = cfd_data[anc1,j,k] - f2[j] - f3[k] - f0
                    
                    #--------------------------------------------------
                    #c_Anova expansion
                    #--------------------------------------------------
                    
                    
                    f_anova[i,j,k] = f0 + f1[i] + f2[j] + f3[k] + f12[i,j] + f13[i,k] + f23[j,k]
                    

        return f_anova