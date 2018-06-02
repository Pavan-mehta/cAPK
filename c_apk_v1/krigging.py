#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:41:42 2018

@author: pmehta
"""


from __future__ import division
import numpy as np
import matplotlib.pyplot as pl


"""
This file performs Gaussion Process Regression or Kriging in One Dimension only

TODO: N - Dimensional Krigging

"""

class kernel:
    
    """
    This class is reserved for definig kernels or co-variance fucntions for Krigging"
    """
    
    def sq_exponential(a, b, theta):
        
        """ GP squared exponential kernel """
        
        if theta is None:
            theta = 0.1
        
        kernelParameter = theta
        
        sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
        
        return np.exp(-.5 * (1/kernelParameter) * sqdist)


    #----------------------------------------------------------------------------

    def poly_cubic_spline(a,b, theta):
        
        """Polynomial Cubic spline kernel: Luca and Sagaut (2016)"""
        
        if theta is None:
            theta = 0.1
                
        def covariance(x1,y1, theta):
            
            """ Covaraince function as per Luca and Sagaut (2016)
                
                The default value of theta = 0.1 as per article 
            
            """
        
            dist = np.absolute(x1-y1)
        
            if dist < (0.5/theta):
                covar = 1 - 6*((dist*theta)**2) + 6*((dist*theta)**3)
            elif (0.5/theta) <= dist < (1/theta):
                covar = 2*((1 - (dist*theta))**3)
            elif dist >= (1/theta):
                covar = 0
            else:
                raise "Value Error: Check Cubic Spline Kernel"
                
            return covar
        
        
        #Computing Ranges 
        
        S1, S2 = a.shape, b.shape
        
        R_1, R_2 = S1[0], S2[0]
        
        
        # Buliding Kernel matix
        
        kernel = np.zeros((R_1,R_2))
                       
        for i in range(R_1):
            for j in range(R_2):
                kernel[i,j] = covariance(a[i],b[j], theta)
     
        
        return kernel


#---------------------------------------------------------------------------------

class gaussion_process_regression:
    
    """ This class is reserved for performing krigging """
    
    def fit(X,Y, Kernel = None, n_points = 100, theta = 0.1):
        
        """ 
        Krigging in 2D: X and Y = F(X) 
        
        Important variables:
            X: For x values, Type: 1D array
            Y: For F(x) values, Type: 1D array
            n_points: Specifiying the number of discretisataion points: Type: int
            discrestisation: array storing the discretised points
            variance: array storing varainaces at discretised points
            mu: array storing mean value at discretised points
        
        """
        
        #For robustness
        X = X.reshape(-1,1)
        Y = Y.reshape(-1,1)
        
        #Range of X or dataset
        R_1 = min(X) 
        R_2 = max(X)
        
        # points we're going to make predictions at.
        discrestisation = np.linspace(R_1,R_2, n_points).reshape(-1,1)
        
                
        #Kernel Selection----------------------------------------------------------
        
        if (Kernel is None) or (Kernel == "sq_exponential"): #use square kernel
                K = kernel.sq_exponential(X, X, theta)
                
                K1 = kernel.sq_exponential(X, discrestisation, theta)
                
                K2 = kernel.sq_exponential(discrestisation, discrestisation, theta)
        
        elif (Kernel == "poly_cubic_spline"):
            
                K = kernel.poly_cubic_spline(X,X, theta)
                                 
                K1 = kernel.poly_cubic_spline(X,discrestisation, theta)
                  
                K2 = kernel.poly_cubic_spline(discrestisation,discrestisation, theta)
                
        else:
            
            raise "Kernel selection Error: Permited value -> 'None', 'sq_exponential', 'poly_cubic_spline' "
        
                       
        #Cholesky decompisition---------------------------------------------------
        L = np.linalg.cholesky(K)
        
                       
        # compute the mean at our test points -----------------------------------
        
        Lk = np.linalg.solve(L, K1)
       
        mu = np.dot(Lk.T, np.linalg.solve(L, Y))
        
        
        # compute the variance at our test points---------------------------------
        
        variance = np.diag(K2) - np.sum(Lk**2, axis=0)
        
                        
        return discrestisation, mu, variance


#---------------------------------------------------------------------------------

class test:
    
    """This class is reserved for testing fucntions only"""
      
    
    def plot(X,Y, Kernel = None, n_points = 100, theta = 0.1):
    
        """ 
        Plot: For visual testing
        
        
        Important variables:
            X: For x values, Type: 1D array
            Y: For F(x) values, Type: 1D array
            n_points: Specifiying the number of discretisataion points: Type: int
            discrestisation: array storing the discretised points
            variance: array storing varainaces at discretised points
            mu: array storing mean value at discretised points
                
        """
        
        #Calling fucntion for perfroming Kriggig        
        discrestisation, mu, variance = gaussion_process_regression.fit(X,Y, Kernel, n_points, theta)
        
        
        #Prepoarting for ploting
        mu = mu.reshape(-1)
        variance = variance.reshape(-1)
        
        #Creating a plot
        pl.figure(1)
        pl.clf()
        pl.plot(X, Y, 'r+', ms=20)
        pl.plot(discrestisation, mu, 'b-')
        pl.gca().fill_between(discrestisation.flat, mu-3*variance, mu+3*variance, color="#dddddd")
        pl.savefig('predictive.png', bbox_inches='tight')
        pl.title('Mean predictions plus 3 st.deviations')
        
        return pl.figure(1)
            
        
