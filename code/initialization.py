#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   initialization.py
@Time    :   2021/12/11 17:21:18
@Author  :   Lin Junwei
@Version :   1.0
@Desc    :   None
'''
#%% import 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import timeit

#%% object function class
def self_generate_cluster(n=100, sigma=1, c = [1,1]):
    """
    Parameters 
    ----------
    n    : size 
    sigma: variance
    c    : centroid 
    Returns
    ----------
    c1   : np.array of location pairs
    """
    c_tuple = tuple(c[i] + np.random.normal(0, sigma, n) for i in range(0,len(c)))
    c1 = np.column_stack(c_tuple)
    return c1





class ObjFunc():
    '''Objective Function Class'''
    def __init__(self, X, a, delta = 1e-3, lam = 1):
        self.X     = X
        self.a     = a
        self.delta = delta
        self.lam   = lam 

    def norm_sum_squ(self, a,x=0,squ=True):
        """
        - Faster way to cal sum of l2 norms

        Parameters 
        ----------
        x  : current centorid X (n*d) # x could be a scalar
        a  : list of points   a (n*d) 

        squ: True  - sum of squared norms (first item in obj func)
            False - sum of simple norms   
        Returns
        ----------
        res  : sum of l2-norms' square
        """
        d   = x - a
        if len(d.shape) == 1:
            d = np.array([d])
        if squ:              # sum of squared L2 norm
            res = np.sum(np.einsum('ij,ij->i',d,d))
        elif squ == False:   # sum of L2 norm
            res = np.sum(np.sqrt(np.einsum('ij,ij->i',d,d)))
        return res

    def hub(self,xi,xj):
        """ 
        Huber norm of centorids
        
        Parameters 
        ----------
        xi : centorid i
        xj : centorid j
        delta : default 1e-3
        """
        
        y_norm = self.norm_sum_squ(xi-xj,0,squ=False)
        if y_norm <= self.delta:
            return (1/(2*self.delta))*y_norm**2
        elif y_norm > self.delta:
            return y_norm - self.delta*0.5


    def grad_hub(self, xi, xj):
        '''
        gradient of huber norm
        '''
        y      = xi - xj
        y_norm = self.norm_sum_squ(y,0,squ=False)
        if y_norm <= self.delta:
            return y/self.delta
        elif y_norm > self.delta:
            return y/y_norm


    def hub_sum_pairwise(self):
        '''
        second item of the obj function
        '''
        ls  = len(self.X)
        res = 0
        for i in range(0,ls):
            for j in range(i+1,ls):
                res += self.hub(self.X[i], self.X[j])
        return res


    def grad_hub_sum_pairwise(self):
        '''gradient of the second item of the obj function'''
        ls  = len(self.X)
        res = 0
        for i in range(0,ls):
            for j in range(i+1,ls):
                res += self.grad_hub(self.X[i], self.X[j])
        return res


    def obj_func(self):
        '''objective function'''
        fx = 0.5*self.norm_sum_squ(a=self.a,x=self.X, squ=True) + self.lam*self.hub_sum_pairwise()
        return fx

    def grad_obj_func(self):
        '''gradient of the objective function'''

        grad_fx = np.sum(self.X-self.a) + self.lam*self.grad_hub_sum_pairwise()
        return grad_fx



#%% Test sample

a1 = self_generate_cluster(n=100, sigma=1, c = [1,1])
a2 = self_generate_cluster(n=100,  sigma=2, c = [3,4])
a = np.concatenate((a1,a2),axis=0)

X  = np.array([[0,0] for i in np.arange(200)])


fx = ObjFunc(X = X, a = a, delta=1e-3, lam=1)

