#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   runAGM.py
@Time    :   2022/01/06 23:43:04
@Author  :   Lin Junwei
@Version :   1.0
@Desc    :   None
'''

#%%
from initialization import *
import numpy as np
import pandas as pd
import time
import os
import random


# %% AGM
def AGM(n, lam, delta, x_k, a, if_use_weight, tol, logname='AGM'+time.strftime('%H_%M_%S', time.localtime()), result_fold = 'AGM_'+time.strftime('%H_%M_%S', time.localtime())):
    
    result_path = str(os.getcwd())+ '\\result\\'+result_fold
    
    if not os.path.exists(result_path):    
        os.makedirs(result_path)
        print("--- Result will be: ", result_fold, ' --- ')
    
    if if_use_weight:
        weights = get_weights(a, 5)
        alpha = 1 / (1 + n * lam * weights.max() / delta)
    else:
        weights = None
        # l2 norm correction delta
        alpha = 1 / (1 + n * lam / delta)

    print('--- AGM Initializing ---')
    grad_coef = grad_hub_coef(X)
    pair_coef = pairwise_coef(X, opera='-')

    matrix_config = {
        'gradient': grad_coef,
        'weights': weights,
        'pairwise': pair_coef}
    

    t_k_1 = 1
    iteration = 0
    x_k_1 = x_k
    obj = ObjFunc(x_k, a, delta=delta, mat_config=matrix_config, lam=lam, if_use_weight=if_use_weight)
    grad_x = obj.grad_obj_func()

    logger = my_custom_logger(str(os.getcwd()) + '\\log\\' + logname + '.log')

    print('--- AGM Starting ---')
    while obj.norm_sum_squ(grad_x, squ=False) > tol:
        
        pickle_write(data=x_k, filenm=str(iteration), folder = result_fold)

        t1 = time.time()
        beta_k = (t_k_1 - 1) / (0.5 * (1 + (1 + 4 * t_k_1 ** 2) ** 0.5))
        t_k_1 = 0.5 * (1 + (1 + 4 * t_k_1 ** 2) ** 0.5)
        y_k = x_k + beta_k * (x_k - x_k_1)
        obj = ObjFunc(y_k, a, delta=delta, mat_config=matrix_config, lam=lam, if_use_weight=if_use_weight)
        x_k_1 = x_k
        x_k = y_k - alpha * (obj.grad_obj_func())
        iteration += 1
        obj = ObjFunc(x_k, a, delta=delta, mat_config=matrix_config, lam=lam, if_use_weight=if_use_weight)
        grad_x = obj.grad_obj_func()

        norm_grad = obj.norm_sum_squ(grad_x, squ=False)
        print('iteration: ', iteration,
              # '\nbeta_k: ', beta_k,
              # '\nt_k_1: ', t_k_1,
              # '\ny_k: ', y_k,
              # '\nx_k: ', x_k,
              # '\ngrad: ', grad_x,
              '\nnorm of grad: ', norm_grad,
              # '\nobj_value: ', obj.obj_func(),
              '\ntime consuming: ', time.time() - t1)

        logger.info('iter:'+str(iteration)+',grad:'+str(norm_grad)+',value:'+str(obj.obj_func())+',time:'+str(time.time()-t1))

    return x_k
#%% test AGM
if __name__ == "__main__":
    t1 = time.time()
    # l2 norm correction
    delta = 0.1
    lam   = 0.5
    tol   = 0.1
    # X  = np.array([[1,1], [1,1], [2,2], [3,3]])
    # a = np.array([[1,1],[1,1],[2,2],[2,2]])
    
    '''synthetic dataset'''
    n = [50, 40, 60]
    sigma = [2, 4, 3]
    c = [[1,1], [10,8], [-3,3]]
    a, syn_label = self_dataset(n=n,sigma=sigma,c=c)
    # X = np.array([[5,2] for i in np.arange(sum(n))]) # initial point
    X = a + np.random.randn(len(a), 2)
    x_k = AGM(sum(n), lam, delta, X, a, False, tol, logname='weighted_AGM_delta1e-1_lam0.5_tol1e-1_3c', result_fold = 'weighted_AGM_delta1e-1_lam0.5_tol1e-1_3c')

    
    '''real dataset'''
    # a, label = load_dataset('wine')
    
    # #choose small batch
    # # np.random.seed(111)
    # # random_index = np.random.choice(len(a), 500, replace=False)
    # # a = a[random_index]
    # # label = label[random_index]
    # # del random_index
    
    # X = np.zeros(a.shape)
    # x_k = AGM(a.shape[0], lam, delta, X, a, True, tol, logname='weighted_AGM_delta1e-1_lam1e-1_tol1_wine', result_fold = 'weighted_AGM_delta1e-1_lam1e-1_tol1_wine')
    
    print('time consuming: ', time.time() - t1)

# %%
