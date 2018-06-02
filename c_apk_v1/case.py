#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:19:53 2018

@author: pmehta
"""

import numpy as np
import pandas as pd


#tesing phase defining flies and data frame explicity - code needs improvement

class define_case:
    
    def case1():

        file_0 = r'cavity_t0.csv'
        file_1 = r'cavity_t1.csv'
        file_2 = r'cavity_t2.csv'
        file_3 = r'cavity_t3.csv'
        file_4 = r'cavity_t4.csv'
        file_5 = r'cavity_t5.csv'
        
        
        df_0 = pd.read_csv(file_0)
        df_1 = pd.read_csv(file_1)
        df_2 = pd.read_csv(file_2)
        df_3 = pd.read_csv(file_3)
        df_4 = pd.read_csv(file_4)
        df_5 = pd.read_csv(file_5)
        
        #corodianted
        
        x = df_0.x
        y = df_0.y
        z = df_0.z
        
        x1 = np.sort(np.hstack(set(x)))

        y1 = np.sort(np.hstack(set(y)))
        
        
        #x_velocities
        
        u1_0 = df_0.u1
        u1_1 = df_1.u1
        u1_2 = df_2.u1
        u1_3 = df_3.u1
        u1_4 = df_4.u1
        u1_5 = df_5.u1
        
        U1 = np.stack((u1_0, u1_1, u1_2, u1_3, u1_4, u1_5), axis = 1)
        
        S1 = (x.shape[0])/2
        S1 = int(S1**0.5)
        
        S2 = (y.shape[0])/2
        S2= int(S2**0.5)
        
        S3 = 5 #max time
        
        cfd_data = np.zeros((S1,S2,S3+1))
        
                
        for k in range(S3):
            a = 0
            for i in range(S1):
                for j in range(S2):
                    cfd_data[i,j,k] = U1[a,k]
                    a += 1


        return cfd_data, U1, x1, y1