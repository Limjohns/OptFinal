#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   runNCG.py
@Time    :   2022/01/07 00:05:37
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

# %% Newton-CG method
def cg(obj, grad, tol, maxiter=10):
    # initialize iterating parameters
    iternum = 0
    grad = grad.reshape(-1, 1)  # (nd, 1) grad
    x = np.zeros((obj.X.size, 1))  # (nd, 1) x0
    r = grad  # (nd, 1) r0
    p = -r  # (nd, 1) p0
    hess_pairwise = obj.hess_hub_pairwise() #(nd, nd) hess matrix of xi-xj
    
    # iterate
    while obj.norm_sum_squ(r, squ=False) > tol and iternum <= maxiter:
        iternum += 1
        hess_prod_p = obj.hess_product_p(hess_pairwise, p)  # (nd, 1) A*p0
        # check if pAp is positive to ensure the correctness of following calculations
        if np.dot(p.T, hess_prod_p) <= 0:
            break
        else:
            r0_norm = obj.norm_sum_squ(r, squ=True)  # scalar norm of r0
            alpha = -r0_norm / (np.dot(p.T, hess_prod_p))[0][0]  # scalar  alpha0
            x = x + alpha * p  # (nd, 1) x1
            r = r + alpha * hess_prod_p  # (nd, 1) r1
            beta = obj.norm_sum_squ(r, squ=True) / r0_norm  # scalar  beta1
            p = -r + beta * p  # (nd, 1) p1
    if iternum == 0:
        return -grad.reshape(obj.X.shape)
    else:
        return x.reshape(obj.X.shape)


def direction_check(d, grad):
    d = d.reshape(-1, 1)
    grad = grad.reshape(-1, 1)
    if np.dot(grad.T, d) < 0:
        return True
    else:
        return False


def newton_cg(obj, s, sigma, gamma, tol, config, result_fold='NCG_'+time.strftime('%H_%M_%S', time.localtime()), logname = 'NCG_'+time.strftime('%H_%M_%S', time.localtime())):
    # Create res pickle path folder
    result_path = str(os.getcwd())+ '\\result\\' + result_fold
    if not os.path.exists(result_path):    
        os.makedirs(result_path)
        print("--- Result will be: ", result_fold, ' --- ')
    # Create log handle
    logger_nw = my_custom_logger(str(os.getcwd()) + '\\log\\' + logname + '.log')


    iteration   = 0
    grad_x      = obj.grad_obj_func()
    grad_x_norm = obj.norm_sum_squ(grad_x, squ=False)

    print('--- Newton CG Starting ---')
    while grad_x_norm > tol and iteration < 5000:
        
        pickle_write(data = obj.X, filenm=str(iteration), folder = result_fold)
       
        iteration += 1
        t1 = time.time()
        # CG direction d
        cg_tol = min(1, grad_x_norm ** 0.1) * grad_x_norm
        d = cg(obj, grad=grad_x, tol=cg_tol)
        # check if is descent direction
        if direction_check(d, grad_x):
            # use CG solutions as direction
            print('cg direction is used')
            pass
        else:
            d = -grad_x

        # choose step size
        # alpha_bck, obj2  = armijo(d, obj, s=s, sigma=sigma, gamma=gamma)
        alpha, obj  = armijo(d, obj, s=s, sigma=sigma, gamma=gamma, config=config)
        # alpha_L = obj.delta / (1 + len(obj.X)*obj.lam)
        # alpha = obj.delta / (1 + len(obj.X)*obj.lam)
        # alpha = max(alpha_bck, alpha_L)
        # obj = ObjFunc(X=obj.X+alpha*d, a=obj.a, grad_coef=obj.grad_coef, delta=obj.delta, lam=obj.lam, if_use_weight=obj.if_use_weight)

        # update iterating parameters
        # obj = ObjFunc(X=obj.X+alpha*d, a=obj.a, grad_coef=obj.grad_coef, delta=obj.delta, lam=obj.lam, if_use_weight=obj.if_use_weight)
        grad_x = obj.grad_obj_func()
        grad_x_norm = obj.norm_sum_squ(grad_x, squ=False)

        print(
            "Iteration:", iteration,
            "\nnorm of grad:", grad_x_norm,
            "\nalpha:", alpha,
            # "\nd:",            d,
            "\ntime: ", time.time() - t1
        )

        logger_nw.info('iter:'+str(iteration)+',grad:'+str(grad_x_norm)+
                       ',alpha:'+str(alpha)+',time:'+str(time.time()-t1))

    return obj.X


# %% test Newton-CG
if __name__ == "__main__":
    t1 = time.time()
    
    '''synthetic dataset'''
    # n = [50, 40, 60]
    # sigma = [2, 4, 3]
    # c = [[1,1], [10,8], [-3,3]]
    # a, syn_label = self_dataset(n=n,sigma=sigma,c=c)
    # # X = np.array([[5,2] for i in np.arange(sum(n))]) # initial point
    # X = a + np.random.randn(len(a), 2)

    '''real datasets'''
    a, label = load_dataset('wine')
    
    #choose small batch
    # np.random.seed(111)
    # random_index = np.random.choice(len(a), 500, replace=False)
    # a = a[random_index]
    # label = label[random_index]
    # del random_index
    X = np.zeros(a.shape)

    if_use_weight = False
    if if_use_weight:
        weights   = get_weights(a, 5)
    else:
        weights = None

    grad_coef = grad_hub_coef(X)
    pair_coef = pairwise_coef(X, opera = '-') 

    matrix_config = {
        'gradient' : grad_coef, 
        'weights'  : weights, 
        'pairwise' : pair_coef}
    

    f = ObjFunc(X=X, a=a, mat_config = matrix_config, delta=1e-1, lam=0.1, if_use_weight=False)
    x_k = newton_cg(obj          =f
                    , s          =1
                    , sigma      =0.5
                    , gamma      =0.1
                    , tol        =1
                    , config     =matrix_config
                    , logname    ='bck_NCG_delta0.1_lam0.1_tol1_wine'
                    , result_fold='bck_NCG_delta0.1_lam0.1_tol1_wine')


    print('time consuming: ', time.time()-t1)
