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
from initialization import load_dataset, self_generate_cluster, grad_hub_coef, self_dataset, get_weights
from initialization import ObjFunc
import numpy as np 
import pandas as pd
import time
import logging
import pickle
import os 


#%% 
def norm_dist(a_i, a_j=0):
    d = a_i - a_j
    if len(d.shape) == 1:
        d = np.array([d])
    res = np.sum(np.einsum('ij,ij->i',d,d))
    return res

def triangular(a):
    n = int(np.sqrt(len(a)*2))+1
    mask = np.arange(n)[:,None] < np.arange(n)
    out = np.zeros((n,n),dtype=np.float32)
    out[mask] = a
    return out

def get_full_dist_array(a):
    weights_list = []
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            weight_ij = norm_dist(a[i], a[j])
            # weight_ij = int(str(i) + str(j))
            weights_list.append(weight_ij)
    weights = triangular(weights_list)
    return weights.T

from scipy.spatial import distance_matrix


mat_dist1 = pd.DataFrame(distance_matrix(ss13, ss13))

mat_dist = pd.DataFrame(get_full_dist_array(ss13))

import timeit 

mat_dist1 = pd.DataFrame(distance_matrix(ss13, ss13))



#%%
def my_custom_logger(logger_name, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s | %(levelname)s | %(message)s")
    log_format = logging.Formatter(fmt=format_string,datefmt='%Y-%m-%d | %H:%M:%S')
    # Creating and adding the console handler
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(log_format)
    # logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def pickle_write(data, filenm, folder='AGM1'):
    with open('result/' + folder + '/' + filenm + ".pkl", "wb") as f:
        pickle.dump(data, f)

def pickle_read(filenm, folder='AGM1'):
    with open('result/' + folder + '/' + filenm + ".pkl", "rb") as f:
        out = pickle.load(f)
    return out

def log_read(logname = 'AGM'):
    path = str(os.getcwd()) + '\\log\\' + logname + '.log'
    with open(path) as f:
        records = []
        for line in f.readlines():
            ls = line.split(' | ')
            records.append(ls[-1].strip())
        all_rec = []
        for rec in records:
            iter_rec = rec.split(',')
            iter_rec = [rec.split(":")[-1] for rec in iter_rec]
            all_rec.append(iter_rec)
    df = pd.DataFrame(all_rec)
    df.columns = ['iteration','Norm_grad','Obj_val','time_consuming']
    return pd.DataFrame(all_rec)


def armijo(d, obj, s, sigma, gamma):
    alpha = s
    obj_2 = ObjFunc(X=obj.X+alpha*d
                    ,a=obj.a
                    ,grad_coef=obj.grad_coef
                    ,delta=obj.delta
                    ,lam=obj.lam
                    ,if_use_weight=obj.if_use_weight)
    while obj_2.obj_func() > obj.obj_func()+gamma*alpha*(np.dot((-obj.grad_obj_func().reshape(-1, 1).T), d.reshape(-1, 1)))[0][0]:
        alpha = alpha * sigma
        obj_2 = ObjFunc(X=obj.X+alpha*d
                        ,a=obj.a
                        ,grad_coef=obj.grad_coef
                        ,delta=obj.delta
                        ,lam=obj.lam
                        ,if_use_weight=obj.if_use_weight)    
    return alpha, obj_2

#%% Newton-CG method
def cg(obj, grad, tol, maxiter=10):
    # initialize iterating parameters
    iternum = 0
    grad        = grad.reshape(-1,1)                                     #(nd, 1) grad
    x           = np.zeros((obj.X.size, 1))                              #(nd, 1) x0
    # hess_prod_x = obj.hess_product_p(x)                                #(nd, 1) A*x0
    r           = grad                                                   #(nd, 1) r0
    p           = -r                                                     #(nd, 1) p0
    
    # iterate
    while obj.norm_sum_squ(r, squ=False) > tol and iternum <= maxiter:
        iternum += 1
        hess_prod_p = obj.hess_product_p(p)                                  #(nd, 1) A*p0
        # check if pAp is positive to ensure the correctness of following calculations
        if np.dot(p.T, hess_prod_p) <= 0:
            break
        else:
            r0_norm = obj.norm_sum_squ(r, squ=True)                   #scalar norm of r0
            alpha   = -r0_norm / (np.dot(p.T, hess_prod_p))[0][0]     #scalar  alpha0
            x       = x + alpha * p                                   #(nd, 1) x1
            r       = r + alpha * hess_prod_p                         #(nd, 1) r1
            beta    = obj.norm_sum_squ(r, squ=True) / r0_norm         #scalar  beta1
            p       = -r + beta * p                                   #(nd, 1) p1
    if iternum == 0:
        return -grad.reshape(obj.X.shape)
    else:
        return x.reshape(obj.X.shape)


def direction_check(d, grad): 
    d    = d.reshape(-1, 1)
    grad = grad.reshape(-1, 1)
    if np.dot(grad.T, d) < 0:
        return True
    else:
        return False

def newton_cg(obj, s, sigma, gamma, tol):
    iteration   = 0
    grad_x      = obj.grad_obj_func()
    grad_x_norm = obj.norm_sum_squ(grad_x, squ=False)
    
    while grad_x_norm > tol and iteration < 5000:
        iteration += 1
        
        # CG direction d
        cg_tol = min(1, grad_x_norm**0.1) * grad_x_norm
        d      = cg(obj, grad=grad_x, tol=cg_tol)
        # check if is descent direction
        if direction_check(d, grad_x):
            # use CG solutions as direction
            print('cg direction is used')
            pass
        else:
            d = -grad_x
            
        # choose step size
        alpha, obj  = armijo(d, obj, s=s, sigma=sigma, gamma=gamma)
        
        # update iterating parameters
        grad_x      = obj.grad_obj_func()
        grad_x_norm = obj.norm_sum_squ(grad_x, squ=False)
        
        print(  
            "Iteration:",    iteration, 
            "\nnorm of grad:", grad_x_norm,
            )
        
    return obj.X
#%% test Newton-CG
if __name__ == "__main__":
    t1 = time.time()
    X  = np.array([[1,1], [1,1], [2,2], [3,3]])
    a = np.array([[1,1],[1,1],[2,2],[2,2]])
    coef = grad_hub_coef(X)
    f = ObjFunc(X = X, a = a, grad_coef=coef, delta=1e-3, lam=0.05, if_use_weight=False)
    x_k = newton_cg(obj=f, s=1, sigma=0.5, gamma=0.1, tol=1e-3)
    # n1 = 100
    # n2 = 100
    # a, syn_label = self_dataset(n1=n1,n2=n2,sigma1=1,sigma2=2,c1=[1,1],c2=[3,3])
    # X = np.array([[2,2] for i in np.arange(n1+n2)]) # initial point
    # coef = grad_hub_coef(X)
    # x_k = AGM(n1+n2, lam, delta, X, a, coef, False, tol)
    # f = ObjFunc(X=X, a=a, grad_coef=coef, delta=delta, lam=lam, if_use_weight=False)
    print('time consuming: ', time.time()-t1)


#%% accelerated gradient method 
def AGM(n, lam, delta, x_k, a, coef, if_use_weight, tol, logname='AGM'):
    if if_use_weight:
        weights = get_weights(a, 5)
    else:
        weights = None
    alpha     = 1/(1+n*lam/delta)
    t_k_1     = 1
    iteration = 0
    x_k_1     = x_k
    obj       = ObjFunc(x_k, a, delta = delta, grad_coef=coef, weights_mat=weights, lam = lam, if_use_weight = if_use_weight)
    grad_x    = obj.grad_obj_func()
    
    
    logger = my_custom_logger(str(os.getcwd()) + '\\log\\' + logname + '.log')

    while obj.norm_sum_squ(grad_x, squ=False) > tol:
        
        pickle_write(data=x_k, filenm=str(iteration))

        t1         = time.time()
        beta_k     = (t_k_1-1)/(0.5*(1+(1+4*t_k_1**2)**0.5))
        t_k_1      = 0.5*(1+(1+4*t_k_1**2)**0.5)
        y_k        = x_k + beta_k*(x_k - x_k_1)
        obj        = ObjFunc(y_k, a, delta=delta, grad_coef=coef, weights_mat=weights, lam=lam, if_use_weight=if_use_weight)
        x_k_1      = x_k
        x_k        = y_k - alpha*(obj.grad_obj_func())
        iteration += 1
        obj        = ObjFunc(x_k, a, delta=delta, grad_coef=coef, weights_mat=weights, lam=lam, if_use_weight=if_use_weight)
        grad_x     = obj.grad_obj_func()

        
        norm_grad  = obj.norm_sum_squ(grad_x, squ=False)
        print('iteration: ', iteration,
              # '\nbeta_k: ', beta_k,
              # '\nt_k_1: ', t_k_1,
              # '\ny_k: ', y_k,
              # '\nx_k: ', x_k,
              # '\ngrad: ', grad_x,
              '\nnorm of grad: ', norm_grad,
              '\nobj_value: ', obj.obj_func(),
              '\ntime consuming: ', time.time()-t1)
        
        logger.info('iter:'+str(iteration)+',grad:'+str(norm_grad)+',value:'+str(obj.obj_func())+',time:'+str(time.time()-t1))
        
    return x_k
#%% test AGM
if __name__ == "__main__":
    t1 = time.time()
    delta = 1e-3 
    lam   = 0.05
    tol   = 1
    # X  = np.array([[1,1], [1,1], [2,2], [3,3]])
    # a = np.array([[1,1],[1,1],[2,2],[2,2]])
    # coef = grad_hub_coef(X)
    # f = ObjFunc(X = X, a = a, grad_coef=coef, delta=delta, lam=lam, if_use_weight=False)
    # AGM(4, lam, delta, X, a, coef, False, tol)
    n1 = 100
    n2 = 100
    a, syn_label = self_dataset(n1=n1,n2=n2,sigma1=1,sigma2=2,c1=[1,1],c2=[10,10])
    X = np.array([[5,2] for i in np.arange(n1+n2)]) # initial point
    coef = grad_hub_coef(X)
    x_k = AGM(n1+n2, lam, delta, X, a, coef, False, tol, logname='AGM')
    # f = ObjFunc(X=X, a=a, grad_coef=coef, weights_mat=weights, delta=delta, lam=lam, if_use_weight=True)
    print('time consuming: ', time.time()-t1)
    
    '''
    测试 1
    delta = 1e-3
    lam   = 0.001
    tol   = 1e-2
    n1 = 100
    n2 = 100
    收敛，耗时20min，迭代580+次
    
    测试 2
    delta = 1e-3
    lam   = 0.005
    tol   = 1e-2
    n1 = 100
    n2 = 100
    收敛，迭代1711次，耗时3889s
    '''
#%% AGM_armijo
def armijo2(s, sigma, gamma, y_k, a, delta, coef, lam, if_use_weight):
    alpha = s
    obj_1 = ObjFunc(y_k, a, delta=delta, grad_coef=coef, lam=lam, if_use_weight=if_use_weight)
    d = -obj_1.grad_obj_func()
    obj_2 = ObjFunc(y_k+alpha*d, a, delta=delta, grad_coef=coef, lam=lam, if_use_weight=if_use_weight)
    while obj_2.obj_func() > obj_1.obj_func() + gamma*alpha*(np.dot(-d.reshape(-1,1).T, d.reshape(-1, 1))):
        alpha = sigma*alpha
        obj_2 = ObjFunc(y_k + alpha * d, a, delta=delta, grad_coef=coef, lam=lam, if_use_weight=if_use_weight)
    return alpha

def AGM_armijo(lam, delta, x_k, a, coef, if_use_weight, tol, s, sigma, gamma):
    # alpha = s
    t_k_1 = 1
    iteration = 0
    x_k_1 = x_k
    obj = ObjFunc(x_k, a, delta = delta, grad_coef=coef, lam = lam, if_use_weight = if_use_weight)
    grad_x = obj.grad_obj_func()

    while obj.norm_sum_squ(grad_x, squ=False) > tol:
        t1 = time.time()
        beta_k = (t_k_1-1)/(0.5*(1+(1+4*t_k_1**2)**0.5))
        t_k_1 = 0.5*(1+(1+4*t_k_1**2)**0.5)
        y_k = x_k + beta_k*(x_k - x_k_1)
        obj = ObjFunc(y_k, a, delta=delta, grad_coef=coef, lam=lam, if_use_weight=if_use_weight)
        x_k_1 = x_k
        alpha = armijo2(s, sigma, gamma, y_k, a, delta, coef, lam, if_use_weight)
        x_k = y_k - alpha*(obj.grad_obj_func())
        iteration += 1
        obj = ObjFunc(x_k, a, delta=delta, grad_coef=coef, lam=lam, if_use_weight=if_use_weight)
        grad_x = obj.grad_obj_func()
        print('iteration: ', iteration,
              # '\nbeta_k: ', beta_k,
              # '\nt_k_1: ', t_k_1,
              # '\ny_k: ', y_k,
              # '\nx_k: ', x_k,
              # '\ngrad: ', grad_x,
              '\nnorm of grad: ', obj.norm_sum_squ(grad_x, squ=False),
              '\nobj_value: ', obj.obj_func(),
              '\ntime consuming: ', time.time()-t1)

    return x_k