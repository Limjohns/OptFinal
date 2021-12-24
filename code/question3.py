# -*- coding: utf-8 -*-
'''
@File    :   question3.py
@Time    :   2021/12/13 15:05:37
@Author  :   Zheng Mengchu
@Version :   1.0
@Desc    :   processing file for question3
'''
#%%
from initialization import load_dataset, self_dataset
from initialization import ObjFunc
import numpy as np 
import pandas as pd 

#%% globalized newton method

def direction_check(d, grad_x, beta1=1e-6, beta2=0.1):
    test1 = np.matmul(grad_x.T, d)
    if test1 > 0:
        return False
    else:
        test2 = -test1
        expected2 = beta1 * min(1, np.linalg.norm(d,2)**beta2) * np.linalg.norm(d,2)**2
        if test2 < expected2:
            return False
        else:
            return True

def newton_glob(obj, s, sigma, gamma, tol):
    
    
    # obj = f
    # s = 1
    # sigma = 0.5
    # gamma = 1e-4
    # tol = 1e-3
    
    
    
    
    iteration = 0
    grad_x = obj.grad_obj_func()
    hess_x = obj.hess_obj_func()
    
    while obj.norm_sum_squ(grad_x, squ=False) > tol and iteration < 5000:
        iteration += 1
        print(iteration)
        
        # choose direction
        d = -np.linalg.solve(hess_x, grad_x) # d = -grad_x / hess_x
        
        if direction_check(d, grad_x):
            print('Newton direction is used')
            pass
        else:
            print('Newton direction is not used')
            d = -grad_x
            
        # choose step size
        alpha = s
        y       = obj.obj_func()
        f_alpha = ObjFunc(X=obj.X+alpha*d, a=obj.a, delta=1e-3, lam=1, if_use_weight=True)
        y_alpha = f_alpha.obj_func()
        # print(y_alpha, y)
        i=0
        while (y_alpha - y) > (gamma * alpha * np.matmul(grad_x.T, d)) and i<50:
            i+=1
            print('inininin')
            alpha = alpha * sigma
            f_alpha = ObjFunc(X=obj.X+alpha*d, a=obj.a, delta=1e-3, lam=1, if_use_weight=True)
            y_alpha = f_alpha.obj_func()
            print(y_alpha)
        print('The stepsize is:', alpha)
        
        # update iterating parameters
        X = obj.X + alpha*d
        obj = f_alpha
        grad_x = obj.grad_obj_func()
        hess_x = obj.hess_obj_func()
        # print(grad_x)
        # print(obj.norm_sum_squ(grad_x, squ=False))
        print(X)
        
    
    return X


#%% prepare data
# a = load_dataset(dataset='wine')
syn_data, syn_label = self_dataset(n1=10,n2=8,sigma1=1,sigma2=2,c1=[1,1],c2=[3,3])

X  = np.array([[0,0] for i in np.arange(18)]) # initial point


fx = ObjFunc(X=X, a=syn_data, delta=1e-3, lam=1, if_use_weight=True)
newton_glob(fx, s=1000, sigma=0.5, gamma=0.1, tol=1e-3)

#%% test
X  = np.array([[0,0] for i in np.arange(4)])
a = np.array([[1,1],[1,1],[2,2],[2,2]])
f = ObjFunc(X = X, a = a, delta=1e-3, lam=1, if_use_weight=True)


newton_glob(f, s=1, sigma=0.5, gamma=1e-4, tol=1e-3)

# %%
