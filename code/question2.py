#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   question2.py
@Time    :   2021/12/13 15:05:37
@Author  :   Lin Junwei
@Version :   1.0
@Desc    :   processing file for question2
'''

#%%
from initialization import load_dataset, self_generate_cluster
from initialization import ObjFunc
import numpy as np 
import pandas as pd 
#%% 
# a = load_dataset(dataset='wine')
a1 = self_generate_cluster(n=100, sigma=1, c = [1,1])
a2 = self_generate_cluster(n=100,  sigma=2, c = [3,4])
a = np.concatenate((a1,a2),axis=0)

X  = np.array([[0,0] for i in np.arange(200)])


fx = ObjFunc(X = X, a = a, delta=1e-3, lam=1)
fx.obj_func()
# %% accelerated gradient method 



#%% Newton-CG method
def cg(obj, grad, tol, maxiter=10):
    # initialize iterating parameters
    iternum = 0
    x           = np.zeros((obj.X.size, 1))                              #(nd, 1) x0
    hess_prod_x = obj.hess_product_p(x)                                  #(nd, 1) A*x0
    r           = obj.hess_product_p(np.zeros((obj.X.size, 1))) + grad   #(nd, 1) r0
    p           = -r                                                     #(nd, 1) p0
    
    # iterate
    while obj.norm_sum_squ(r, squ=False) > tol and iternum <= maxiter:
        iternum += 1
        hess_prod_p = obj.hess_product_p(p)                                        #(nd, 1) A*p0
        # check if pAp is positive to ensure the correctness of following calculations
        if np.dot(p.T, hess_prod_p) <= 0:
            break
        else:
            alpha       = -(np.dot(r.T, p)) / (np.dot(p.T, hess_prod_p))           #scalar  alpha0
            x           = x + alpha * p                                            #(nd, 1) x1
            r           = r + alpha * hess_prod_p                                  #(nd, 1) r1
            beta        = (np.dot(r.T, hess_prod_p)) / (np.dot(p.T, hess_prod_p))  #scalar  beta1
            p           = -r + beta * p                                            #(nd, 1) p1
    if iternum == 0:
        return -grad
    else:
        return x

def direction_check(d):
    



def newton_glob(obj, s, sigma, gamma, tol):
  
    iteration = 0
    grad_x = obj.grad_obj_func()
    hess_x = obj.hess_obj_func()
    
    while obj.norm_sum_squ(grad_x, squ=False) > tol and iteration < 5000:
        iteration += 1
        print(iteration)
        
        # CG direction d
        grad_x_norm = obj.norm_sum_squ(grad_x, squ=False)
        cg_tol = min(1, grad_x_norm**0.1) * grad_x_norm
        d = cg(obj, grad=grad_x, tol=cg_tol)
        
        
        if direction_check(d, grad_x):
            # use CG solutions as direction
            pass
        else:
            # use backup direction 'v'
            
        # choose step size
        alpha = s
        y       = obj.obj_func()
        f_alpha = ObjFunc(X=obj.X+alpha*d, a=obj.a, delta=1e-3, lam=1, if_use_weight=True)
        y_alpha = f_alpha.obj_func()
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