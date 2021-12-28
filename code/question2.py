#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   question2.py
@Time    :   2021/12/13 15:05:37
@Author  :   Lin Junwei
@Version :   1.0
@Desc    :   processing file for question2
'''

# %%
from initialization import load_dataset, self_generate_cluster, grad_hub_coef, self_dataset, get_weights, pairwise_coef
from initialization import ObjFunc
import numpy as np
import pandas as pd
import time
import logging
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import manifold
#%%

# %%
def my_custom_logger(logger_name, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level) 
    format_string = ("%(asctime)s | %(levelname)s | %(message)s")
    log_format = logging.Formatter(fmt=format_string, datefmt='%Y-%m-%d | %H:%M:%S')
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
    with open(os.path.join(r'/Users/leongvan/Documents/GitHub/OptFinal/code/result', folder, "{0}.pkl".format(filenm)), "wb") as f:
        pickle.dump(data, f)


def pickle_read(filenm, folder='AGM1'):
    with open('result/' + folder + '/' + filenm + ".pkl", "rb") as f:
        out = pickle.load(f)
    return out


def log_read(logname='AGM'):
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
    df.columns = ['iteration', 'Norm_grad', 'Obj_val', 'time_consuming']
    return pd.DataFrame(all_rec)


def armijo(d, obj, s, sigma, gamma,config):
    alpha = s
    obj_2 = ObjFunc(X           = obj.X+alpha*d
                   ,a          = obj.a
                   ,mat_config = config
                   ,delta      = obj.delta
                   ,lam        = obj.lam
                   ,if_use_weight = obj.if_use_weight)
    while obj_2.obj_func() > obj.obj_func() + gamma * alpha * \
            (np.dot((obj.grad_obj_func().reshape(-1, 1).T), d.reshape(-1, 1)))[0][0]:
        alpha = alpha * sigma
        obj_2 = ObjFunc(X           = obj.X+alpha*d
                        ,a          = obj.a
                        ,mat_config = config
                        ,delta      = obj.delta
                        ,lam        = obj.lam
                        ,if_use_weight = obj.if_use_weight)  
    return alpha, obj_2


# %% Newton-CG method
def cg(obj, grad, tol, maxiter=10):
    # initialize iterating parameters
    iternum = 0
    grad = grad.reshape(-1, 1)  # (nd, 1) grad
    x = np.zeros((obj.X.size, 1))  # (nd, 1) x0
    # hess_prod_x = obj.hess_product_p(x)                                #(nd, 1) A*x0
    r = grad  # (nd, 1) r0
    p = -r  # (nd, 1) p0

    # iterate
    while obj.norm_sum_squ(r, squ=False) > tol and iternum <= maxiter:
        iternum += 1
        hess_prod_p = obj.hess_product_p(p)  # (nd, 1) A*p0
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

# if __name__ == "__main__":
#     t1 = time.time()
#     # X  = np.array([[1,1], [1,3], [2,2], [3,3]])
#     # a = np.array([[1,1],[1,6],[2,2],[2,2]])
#     # coef = grad_hub_coef(X)
#     # f = ObjFunc(X = X, a = a, grad_coef=coef, delta=1e-3, lam=0.05, if_use_weight=False)
#     # x_k = newton_cg(obj=f, s=1, sigma=0.5, gamma=0.1, tol=1e-2)
#     n1 = 50
#     n2 = 50
#     a, syn_label = self_dataset(n1=n1,n2=n2,sigma1=1,sigma2=1,c1=[1,1],c2=[10,10])
#     X = np.array([[5,2] for i in np.arange(n1+n2)]) # initial point
#     # X = a + np.random.randn(len(a), 2)
#     if_use_weight = False
#
#     if if_use_weight:
#         weights   = get_weights(a, 5)
#     else:
#         weights = None
#     grad_coef = grad_hub_coef(X)
#     pair_coef = pairwise_coef(X, opera = '-')
#
#     matrix_config = {
#         'gradient' : grad_coef,
#         'weights'  : weights,
#         'pairwise' : pair_coef}
#
#
#     # coef = grad_hub_coef(X)
#     f = ObjFunc(X=X, a=a, mat_config = matrix_config, delta=1e-3, lam=0.05, if_use_weight=False)
#
#     x_k = newton_cg(obj=f, s=1, sigma=0.5, gamma=0.1, tol=1, config=matrix_config)
#
#     print('time consuming: ', time.time()-t1)

if __name__ == "__main__":
    t1 = time.time()
    X  = np.array([[1,1], [1,3], [2,2], [3,3]])
    a = np.array([[1,1],[1,6],[2,2],[2,2]])
    # coef = grad_hub_coef(X)
    # f = ObjFunc(X = X, a = a, grad_coef=coef, delta=1e-3, lam=0.05, if_use_weight=False)
    # x_k = newton_cg(obj=f, s=1, sigma=0.5, gamma=0.1, tol=1e-2)
    
    # n = [30, 20, 20]
    # sigma = [2, 5, 4]
    # c = [[1,1], [10,14], [-3,3]]
    # a, syn_label = self_dataset(n=n,sigma=sigma,c=c)

    # X = a + np.random.randn(len(a), 2)

    
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

    
    # coef = grad_hub_coef(X)
    f = ObjFunc(X=X, a=a, mat_config = matrix_config, delta=1e-3, lam=0.05, if_use_weight=False)

    x_k = newton_cg(obj=f, s=1, sigma=0.5, gamma=0.1, tol=0.01, config=matrix_config, result_fold='NCG'+time.strftime('%H_%M_%S', time.localtime()), logname = 'NCG'+time.strftime('%H_%M_%S', time.localtime()))

    print('time consuming: ', time.time()-t1)



# %% accelerated gradient method
def AGM(n, lam, delta, x_k, a, if_use_weight, tol, logname='AGM'+time.strftime('%H_%M_%S', time.localtime()), result_fold = 'AGM_'+time.strftime('%H_%M_%S', time.localtime())):
    
    # result_path = str(os.getcwd())+ '\\result\\'+result_fold
    result_path = os.path.join(r'/Users/leongvan/Documents/GitHub/OptFinal/code/result', result_fold, )
    
    if not os.path.exists(result_path):    
        os.makedirs(result_path)
        print("--- Result will be: ", result_fold, ' --- ')
    
    if if_use_weight:
        weights = get_weights(a, 5)
    else:
        weights = None


    print('--- AGM Initializing ---')
    grad_coef = grad_hub_coef(X)
    pair_coef = pairwise_coef(X, opera='-')

    matrix_config = {
        'gradient': grad_coef,
        'weights': weights,
        'pairwise': pair_coef}
    
    # l2 norm correction delta
    alpha = 1 / (1 + n * lam / delta)
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
    lam   = 0.1
    tol   = 1
    # X  = np.array([[1,1], [1,1], [2,2], [3,3]])
    # a = np.array([[1,1],[1,1],[2,2],[2,2]])
    # coef = grad_hub_coef(X)
    # f = ObjFunc(X = X, a = a, grad_coef=coef, delta=delta, lam=lam, if_use_weight=False)
    # AGM(4, lam, delta, X, a, coef, False, tol)
    
    # random points
    # n = [30, 40, 20]
    # sigma = [2, 5, 4]
    # c = [[1,1], [10,14], [-3,3]]
    # a, syn_label = self_dataset(n=n,sigma=sigma,c=c)
    # # X = np.array([[5,2] for i in np.arange(sum(n))]) # initial point
    # X = a + np.random.randn(len(a), 2)
    # x_k = AGM(sum(n), lam, delta, X, a, False, tol,logname='AGM_1')
    
    
    # wine dataset
    a, label = load_dataset('vowel')
    # a = TSNE(n_components=2,random_state=0,init='pca').fit_transform(a)
    # a = manifold.Isomap(n_components=2).fit_transform(a)
    X = np.zeros(a.shape)
    x_k = AGM(a.shape[0], lam, delta, X, a, False, tol, logname='AGM_12_vowel')
    print('time consuming: ', time.time() - t1)
    


# %% plot AGM
#
# filels = os.listdir(r'C:\Users\Lenovo\Desktop\OptFinal\code\result\AGM1')
# array_ls = []
# for file in filels:
#     file = file.split('.')[0]
#     df = pickle_read(file, folder='AGM1')
#     array_ls.append(df)
#
# def plot_points(arr, label):
#     c = pd.Series(label.reshape(-1,)).apply(lambda x: 'red' if x==1 else('blue' if x==2 else 'green'))
#     pd.DataFrame(arr).plot.scatter(x=0, y=1, c=c)
#     # plt.xlim(-20,15)
#     plt.show()
#
# I = 0
# for arr in array_ls[:2]:
#     print(I, 'th-----------------------')
#     # arr = TSNE(n_components=2,random_state=0,init='pca').fit_transform(arr)
#     # arr = manifold.Isomap(n_components=2).fit_transform(arr)
#     plot_points(arr, label)
#     I +=1
#
#
# # a = TSNE(n_components=2,random_state=0,init='pca').fit_transform(a)
# # a = manifold.Isomap(n_components=2).fit_transform(a)
# plot_points(a, label)

