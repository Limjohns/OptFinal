#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   initialization.py
@Time    :   2021/12/11 17:21:18
@Author  :   Lin Junwei
@Version :   1.0
@Desc    :   initialization class and function
'''
#%% import 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import timeit
import os

#%% load data function

def self_generate_cluster(n=100, sigma=1, c = [1,1]):
    """
    Parameters 
    ----------
    n    : size 
    sigma: variance
    c    : centroid 

    Returns
    ----------
    c1  : np.array in shape (n, dimensions)
    """
    c_tuple = tuple(c[i] + np.random.normal(0, sigma, n) for i in range(0,len(c)))
    c1 = np.column_stack(c_tuple)
    return c1

def self_dataset(n1=100,n2=100,sigma1=1,sigma2=2,c1=[1,1],c2=[3,3]):
    """
    Parameters 
    ---------- 
    n    : size 
    sigma: variance
    c    : centroid 

    Returns
    ----------
    allset.T  : np.array in shape (total samples, features)
    alllabel.T: np.array in shape (total samples, 1)

    """
    set1 = self_generate_cluster(n = n1, sigma = sigma1, c = c1)
    set2 = self_generate_cluster(n = n2, sigma = sigma2, c = c2)
    label1 = np.array([[0 for i in range(0,n1)]])
    label2 = np.array([[1 for i in range(0,n2)]])
    allset = np.concatenate((set1,set2),axis=0)
    alllabel = np.concatenate((label1,label2),axis=1)

    return allset, alllabel.T

def load_dataset(dataset = 'wine'):
    """
    Parameters 
    ----------
    dataset : name of dataset

    Returns
    ----------
    data  : array in shape: (samples, features)
    label : array in shape: (samples, 1)

    """
        
    data_path  = 'datasets/datasets/{}/{}_data.mat'.format(dataset,dataset)
    label_path = 'datasets/datasets/{}/{}_label.mat'.format(dataset,dataset)
    data       = scipy.io.loadmat(data_path)['A']
    label      = scipy.io.loadmat(label_path)['b']

    if dataset != 'mnist':
        data   = data.toarray()
        
    data = data.T
    print(dataset,' - data shape: ',data.shape, '; label shape: ',label.shape)

    return data, label

def grad_hub_coef(X):
    n, d = X.shape
    def loc_fun(row_num, n):
        if row_num == 1:
            return None, 0, n-1
        else:
            lower_tri = np.tril(np.ones((row_num-1, row_num-1)), k=-1)
            n_seque = np.array([n-j for j in range(1, row_num)]).reshape((-1,1))
            loc_coef = np.dot(lower_tri, n_seque)
            # 生成0, n-1, (n-1)+(n-2),...序列
            #生成-1在第i行的位子
            loc_minus_1 = loc_coef - np.arange(row_num-1).reshape((-1,1)) + row_num - 2
            #生成1在第i行的位子
            loc_1_start = (2*n-row_num)*(row_num-1)/2
            loc_1_end = loc_1_start + n - row_num
            return loc_minus_1.astype(np.int16).reshape((-1,)).tolist(), int(loc_1_start), int(loc_1_end)
    rows = []
    for row_num in range(1, n+1):
        loc_minus_1, loc_1_start, loc_1_end = loc_fun(row_num, n)
        row = np.zeros((int(n*(n-1)/2),))
        if loc_minus_1 is None:
            pass
        else:
            row[loc_minus_1] = -1
        row[loc_1_start:loc_1_end] = 1
        rows.append(row)
    return np.stack(rows)

#%%  objective function class
class ObjFunc():
    '''Objective Function Class'''
    def __init__(self, X, a, grad_coef, delta = 1e-3, lam = 1, if_use_weight = False):
        self.X             = X
        self.a             = a
        self.delta         = delta
        self.lam           = lam 
        self.if_use_weight = if_use_weight
        self.grad_coef     = grad_coef

    def norm_sum_squ(self, a,x=0,squ=True):
        """
        Faster way to cal sum of l2 norms

        Parameters 
        ----------
        x  : current centorid X => [n*d]  (x could be a scalar)
        a  : list of points   a => [n*d]  (when a = 0, it will cal the self l2-norm)

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
        xi : centorid i => [1*d]
        xj : centorid j => [1*d]
        delta : huber norm param, default 1e-3
        """
        
        y_norm = self.norm_sum_squ(xi-xj,0,squ=False)
        if y_norm <= self.delta:
            return (1/(2*self.delta))*y_norm**2
        elif y_norm > self.delta:
            return y_norm - self.delta*0.5  #return scalar


    def grad_hub(self, xi, xj=0):
        '''
        Gradient of huber norm 
        
        Parameters 
        ----------
        xi  : Single row vector of X (centorid) => [1*d] 
        xj  : Single row vector of X (centorid) => [1*d] 
        
        Returns
        ----------
        res : Gradient => [1*d] here: (d,)
        '''
        y      = xi - xj
        y_norm = self.norm_sum_squ(y,0,squ=False)
        if y_norm <= self.delta:
            return y/self.delta
        elif y_norm > self.delta:
            return y/y_norm  #return vector

    def hess_hub(self, xi, xj):
        '''
        Hessian of huber norm
        Parameters 
        ----------
        xi  : Single row vector of X => [1*d] 
        xj  : Single row vector of X => [1*d] 
        
        Returns
        ----------
        res : Hessian matrix of Huber(xi - xj) => [d*d]

        '''
        y = xi - xj
        y_norm = self.norm_sum_squ(y,0,squ=False)
        if y_norm <= self.delta:
            return 1/self.delta
        elif y_norm > self.delta:
            return (y_norm**2*np.eye(len(y)) - np.dot(y,y.T)) / y_norm**3  #return matrix
        

    def weight(self, i, j, k=5):
        '''
        Calculate the Weights in the 2nd term
        
        Parameters 
        ----------
        i : index, int
        j : index, int
        k : k nearest neighbors, int
        self.if_use_weight : True when using weights model, Boolean
        
        Returns
        ----------
        1       : When not using weights
        weights : When using weights

        '''

        if self.if_use_weight:
            if abs(i-j) <= k:
                return np.exp(-0.5*self.norm_sum_squ(self.a[i], self.a[j], squ=True))
            else:
                return 0
        else:
            return 1


    def hub_sum_pairwise(self):
        '''
        second item value of the obj function
        
        Returns
        ----------
        res : value
        '''
        ls  = len(self.X)
        res = 0
        for i in range(0,ls):
            for j in range(i+1,ls):
                res += self.hub(self.X[i], self.X[j]) * self.weight(i, j)
        return res

    def partial_grad_hub_sum(self,i):
        '''
        Partial gradient of every rows in the gradient vector
        Parameters 
        ----------
        i : index of variables to be derived, int
        
        Returns
        ----------
        1       - when not using weights
        weights - when using weights

        '''
        partial_grad = 0
        for j in range(0, len(self.X)):
            if j < i:
                partial_grad += -self.grad_hub(self.X[i], self.X[j]) * self.weight(i,j)
            elif j > i:
                partial_grad +=  self.grad_hub(self.X[i], self.X[j]) * self.weight(i,j)

        return partial_grad 


    def grad_hub_sum_pairwise(self):
        '''
        Gradient of the 2nd item of the obj function (vector)
        
        Returns
        ----------
        Gradient: [n*d] (in the same shape with X)

        '''
        return np.array([self.partial_grad_hub_sum(i) for i in range(len(self.X))])  

    def grad_hub_matrix(self):
        '''use matrix to calculate the gradient of the 2nd item'''
        n, d = self.X.shape
        xi_xj = []
        weight_ij = []
        for i in range(n):
            for j in range(i+1,n):
                xi_xj.append(self.X[i]-self.X[j])
                weight_ij.append(self.weight(i,j))
        xi_xj = np.stack(xi_xj)
        weight_ij = np.array(weight_ij).reshape((-1,1))

        grad_xi_xj = np.apply_along_axis(self.grad_hub, 1, xi_xj) * weight_ij
        return np.dot(self.grad_coef, grad_xi_xj)


    def partial_hess_hub_sum(self, i, j):
        '''
        Each element of the Hessian of the second item
        
        Returns
        ----------
        Partial Hessian: [d*d]
        
        '''
        if i == j:
            diagonal_ele = 0
            for k in range(0,len(self.X)):
                if k < i:
                    diagonal_ele += -self.hess_hub(self.X[k], self.X[i]) * self.weight(i,j)
                elif k > i:
                    diagonal_ele +=  self.hess_hub(self.X[i], self.X[k]) * self.weight(i,j)
            return diagonal_ele
        else:
            small = max(i, j)
            large = min(i, j)
            return - self.hess_hub(self.X[small], self.X[large]) * self.weight(i,j)


    # def fill_upper_diag(self, X):
    #     '''
    #     convert a list to upper diagonal
    #     --- 

    #     Example:
    #     ----------
    #     X = [1,2,3,4,5,6]
    #     fill_lower_diag(X) 
    #     ---> array([[0, 1, 2, 3],
    #                 [0, 0, 4, 5],
    #                 [0, 0, 0, 6],
    #                 [0, 0, 0, 0]])

    #     '''
    #     n    = int(np.sqrt(len(X)*2))+1
    #     # mask = np.arange(n)[:,None] < np.arange(n) # or np.tri(n,dtype=bool, k=-1)
    #     mask = np.tri(n,dtype=bool, k=-1)

    #     out  = np.zeros((n,n),dtype=int)
    #     out[mask] = X
    #     return out

    # def hess_hub_sum_pairwise(self):
    #     '''Get the full Hessian Matrix'''
    #     diagnoal  = []
    #     tringular = []
    #     for i in range(0, len(self.X)):
    #         for j in range(i, len(self.X)):
    #             if i != j:
    #                 tringular.append(self.partial_hess_hub_sum(i, j))
    #             else:
    #                 diagnoal.append(self.partial_hess_hub_sum(i, j))
    #     Hess_half = self.fill_upper_diag(tringular)
    #     return Hess_half.T + Hess_half + np.diag(diagnoal)

    def hess_product_p(self, p):
        '''Newton CG A*p_k'''
        n, d = self.X.shape
        for i in range(len(self.X)): # each d rows of vector Hess*d
            for k in range(len(self.X)):  # sum up to calculate each d rows
                if k == 0:
                    hd_i = np.dot(self.partial_hess_hub_sum(i, k), p[k*d : (k+1)*d])
                else:
                    hd_i = np.concatenate((np.dot(self.partial_hess_hub_sum(i, k), p[k*d : (k+1)*d])))
        if i == 0:
            pass
        else:
            pass
        return #np.array(hd).reshape((-1,1)) + p
        

    def obj_func(self):
        '''objective function'''
        fx = 0.5*self.norm_sum_squ(a=self.a,x=self.X, squ=True) + self.lam*self.hub_sum_pairwise()
        return fx

    def grad_obj_func(self):
        '''gradient of the objective function'''

        grad_fx = (self.X-self.a) + self.lam*self.grad_hub_sum_pairwise()
        return grad_fx
    
    # def hess_obj_func(self):
    #     '''Hessian of the objective function'''
    #     first_item_hess  = np.eye(self.X.shape[0]*self.X.shape[1])
    #     second_item_hess = self.hess_hub_sum_pairwise()
    #     return first_item_hess + second_item_hess


#%% Test sample
if __name__ == "__main__":
    a1 = self_generate_cluster(n=5, sigma=1, c = [1,1])
    a2 = self_generate_cluster(n=5,  sigma=2, c = [3,4])
    a = np.concatenate((a1,a2),axis=0)

    X  = np.array([[0,0] for i in np.arange(200)])


    fx = ObjFunc(X = X, a = a, delta=1e-3, lam=1)
    
#%% test
X = np.array([[0,0],[1,2],[3,5],[4,3]])
a = np.array([[1,1],[1,1],[2,2],[2,2]])
coef = grad_hub_coef(X)
f = ObjFunc(X = X, a = a, grad_coef=coef, delta=1e-3, lam=1, if_use_weight=True)

grad = f.grad_hub_sum_pairwise()

# %%
# test
# test main - Lin
