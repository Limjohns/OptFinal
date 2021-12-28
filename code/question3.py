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
import time

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


def AGM(n, lam, delta, x_k, a, if_use_weight, tol):

    alpha = 1/(1+n*lam/delta)
    t_k_1 = 1
    iteration = 0
    x_k_1 = x_k
    obj = ObjFunc(x_k, a, delta = delta, lam = lam, if_use_weight = if_use_weight)
    grad_x = obj.grad_obj_func()

    while obj.norm_sum_squ(grad_x, squ=False) > tol:
        t1 = time.time()
        beta_k = (t_k_1-1)/(0.5*(1+(1+4*t_k_1**2)**0.5))
        t_k_1 = 0.5*(1+(1+4*t_k_1**2)**0.5)
        y_k = x_k + beta_k*(x_k - x_k_1)
        obj = ObjFunc(y_k, a, delta=delta, lam=lam, if_use_weight=if_use_weight)
        x_k_1 = x_k
        x_k = y_k - alpha*(obj.grad_obj_func())
        iteration += 1
        obj = ObjFunc(x_k, a, delta=delta, lam=lam, if_use_weight=if_use_weight)
        grad_x = obj.grad_obj_func()
        print('iteration: ', iteration,
              # '\nbeta_k: ', beta_k,
              # '\nt_k_1: ', t_k_1,
              # '\ny_k: ', y_k,
              # '\nx_k: ', x_k,
              # '\ngrad: ', grad_x,
              '\nnorm of grad: ', obj.norm_sum_squ(grad_x, squ=False),
              '\nobj_value: ', obj.obj_func(),
              '\ntime: ', time.time()-t1)

    return x_k


#%% prepare data
# a = load_dataset(dataset='wine')
# syn_data, syn_label = self_dataset(n1=10,n2=8,sigma1=1,sigma2=2,c1=[1,1],c2=[3,3])
#
# X  = np.array([[0,0] for i in np.arange(18)]) # initial point
#
#
# fx = ObjFunc(X=X, a=syn_data, delta=1e-3, lam=1, if_use_weight=True)
# newton_glob(fx, s=1000, sigma=0.5, gamma=0.1, tol=1e-3)
#
# #%% test
# X  = np.array([[0,0] for i in np.arange(4)])
# a = np.array([[1,1],[1,1],[2,2],[2,2]])
# f = ObjFunc(X = X, a = a, delta=1e-3, lam=1, if_use_weight=True)
#
#
# newton_glob(f, s=1, sigma=0.5, gamma=1e-4, tol=1e-3)

# %%

if __name__ == "__main__":
    t1 = time.time()
    delta = 1e-3
    lam   = 0.005   # 这个参数越大，找得越久；存在的现象是梯度的norm反复横跳，因此这个参数可能影响的是每一步的步长
    tol   = 1e-2
    # X = np.array([[1,1], [1,1], [2,2], [3,3]])
    # X = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    # a = np.array([[1,1], [2,2],[3,3],[4,4]])
    # AGM(4, lam, delta, X, a, False, tol)

    n1 = 100
    n2 = 100
    a, syn_label = self_dataset(n1=n1,n2=n2,sigma1=1,sigma2=2,c1=[1,1],c2=[3,3])
    X = np.array([[0,0] for i in np.arange(n1+n2)]) # initial point
    weights = get_weights(a, 5)
    coef = grad_hub_coef(X)

    x_k = AGM(n1+n2, lam, delta, X, a, False, tol)

    f = ObjFunc(X=X, a=a, delta=delta, lam=lam, if_use_weight=True)
    print('time consuming: ', time.time()-t1)

