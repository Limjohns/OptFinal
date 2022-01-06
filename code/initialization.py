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
from sklearn.cluster import KMeans
import pickle
import os
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import logging


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

def self_dataset(n=[100],sigma=[1],c=[[1,1]]):
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
    for i in range(len(sigma)):
        set_i = self_generate_cluster(n = n[i], sigma = sigma[i], c = c[i])
        label_i = np.array([[i for j in range(0,n[i])]])
        if i  == 0:
            allset = set_i
            alllabel = label_i
        else:
            allset = np.concatenate((allset,set_i),axis=0)
            alllabel = np.concatenate((alllabel,label_i),axis=1)

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

### pairwise difference/add matrix
def pairwise_coef(X, opera = '-'):
    '''
    To Compute the pairwise differences
    By left dot X in shape (n, d) => ((n*(n-1)/2), d)

    Parameter:
    -------- 
    X     : Data, in shape (n,d)      
    opera : str, '-': difference, default
                 '+': cumsum     

    Return: 
    --------
    np.array(row_ls) in shape [(n*(n-1)/2), n]
    '''
    if opera == '-':
        xj_opera = -1
    elif opera == '+':
        xj_opera = 1
    else:
        xj_opera = -1

    n, d = X.shape
    row_ls = []
    for i in range(n):        
        for j in range(i+1, n):
            row    = np.array([0 for i in range(n)])
            row[i] = 1 
            row[j] = -1
            row_ls.append(row)
    return np.array(row_ls)
### Gradient huber matrix
def grad_hub_coef(X):
    '''
    To Compute huber gradient

    By left dot X in shape (n, 1)  
    
    Return: 
    --------
    np.array(row_ls) in shape [(n*(n-1)/2), n]
    '''
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

### Get weights in q3
def get_weights(a, topnum = 5):
    mat_dist = pd.DataFrame(distance_matrix(a, a))
    mat_dist = np.square(mat_dist)
    full_w_arr = np.exp(-0.5*mat_dist).to_numpy()

    def top_k(x,k):
        ind=np.argpartition(x,[i for i in range(k)])[:k]
        return ind[np.argsort(x[ind])]
    
    def weight_topk(weights_arr, top_num = topnum):
        weight_df = pd.DataFrame(weights_arr)
        return np.apply_along_axis(lambda x: top_k(x,top_num+1),0,weight_df.values)[1:]
    
    def weight_mask(weights_arr, top_num=topnum):
        loc = weight_topk(weights_arr, top_num)
        n = len(weights_arr)
        res = np.zeros((n, n))
        for i in range(n):
            res[loc[:,i], i] = weights_arr[loc[:,i], i]
        return res
    
    return weight_mask(full_w_arr, topnum)


### read log and pickle

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
    with open('result/' + folder + '/' + filenm + ".pkl", "wb") as f:
        pickle.dump(data, f)

def pickle_read(filenm, folder='AGM1'):
    with open('result/' + folder + '/' + filenm + ".pkl", "rb") as f:
        out = pickle.load(f)
    return out

def log_read(logname = 'AGM'):
    '''read log to dataframe'''
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
        
        col_name = [rc.split(":")[0] for rc in rec.split(',')]

    df = pd.DataFrame(all_rec)
    df.columns = col_name
    return df



#%% Stepwise Strategy - armijo
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






#%%  objective function class
class ObjFunc():
    '''Objective Function Class'''
    def __init__(self, X, a, mat_config, delta = 1e-3, lam = 1, if_use_weight = False):
        self.X             = X
        self.a             = a
        self.delta         = delta
        self.lam           = lam 
        self.if_use_weight = if_use_weight
        
        self.grad_coef     = mat_config['gradient']
        self.pair_coef     = mat_config['pairwise']

        if self.if_use_weight:
            self.weights_mat = mat_config['weights']
        else: 
            self.weights_mat = np.ones((a.shape[0],a.shape[0]))

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
        # l2 norm correction =
        if y_norm <= self.delta:
            return y/self.delta
        elif y_norm > self.delta:
            return y/y_norm  #return vector

    def hess_hub(self, xi, xj=0):
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
            return np.eye(len(y))/self.delta
        elif y_norm > self.delta:
            return (y_norm**2*np.eye(len(y)) - np.dot(y,y.T)) / y_norm**3  #return matrix
        

    def weight(self, i, j):
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
            return self.weights_mat[i,j]
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

    # def partial_grad_hub_sum(self,i):
    #     '''
    #     Partial gradient of every rows in the gradient vector

    #     Parameters 
    #     ----------
    #     i : index of variables to be derived, int
        
    #     Returns
    #     ----------
    #     1       - when not using weights
    #     weights - when using weights

    #     '''
    #     partial_grad = 0
    #     for j in range(0, len(self.X)):
    #         if j < i:
    #             partial_grad += -self.grad_hub(self.X[i], self.X[j]) * self.weight(i,j)
    #         elif j > i:
    #             partial_grad +=  self.grad_hub(self.X[i], self.X[j]) * self.weight(i,j)

    #     return partial_grad 


    # def grad_hub_sum_pairwise(self):
    #     '''
    #     Gradient of the 2nd item of the obj function (vector)
        
    #     Returns
    #     ----------
    #     Gradient: [n*d] (in the same shape with X)

    #     '''
    #     return np.array([self.partial_grad_hub_sum(i) for i in range(len(self.X))])  

    def grad_hub_matrix(self):
        '''use matrix to calculate the gradient of the 2nd item'''
        
        
        #### new ########################

        xi_xj     = np.dot(self.pair_coef, self.X) 
        weight_ij = self.weights_mat[np.triu_indices(self.X.shape[0], k = 1)].reshape((-1,1))
        
        grad_xi_xj = np.apply_along_axis(self.grad_hub, 1, xi_xj) * weight_ij


        ### old #########################
        # n, d = self.X.shape
        # xi_xj = []
        # weight_ij = []
        # for i in range(n):
        #     for j in range(i+1,n):
        #         xi_xj.append(self.X[i]-self.X[j])
        #         weight_ij.append(self.weight(i,j))
        # xi_xj = np.stack(xi_xj)
        # weight_ij = np.array(weight_ij).reshape((-1,1))
        # grad_xi_xj = np.apply_along_axis(self.grad_hub, 1, xi_xj) * weight_ij
        
        # print(xi_xj)
        # print(weight_ij)
        # print(grad_xi_xj)
        
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


    # def triangular_hess_hub_sum(self):
        
    #     (n,d) = self.X.shape
        
    #     row_hess_list = []
    #     for i in range(0, n):
    #         row_hess = [np.zeros((d, (i+1)*d))]
    #         for j in range(i+1 , n):
    #             row_hess.append(self.partial_hess_hub_sum(i, j))
    #         row_hess_list.append(np.concatenate(row_hess,axis = 1))
    #     full_mat = np.concatenate(row_hess_list)
        
    #     return full_mat


    # def hess_hub_sum_pairwise(self):
    #     '''Get the full Hessian Matrix'''
    #     diagnoal  = []
    #     for i in range(0, len(self.X)):
    #         for j in range(i, len(self.X)):
    #             if i != j:
    #                 pass 
    #             else:
    #                 diagnoal.append(self.partial_hess_hub_sum(i, j))
    #     diagnoal = np.concatenate(diagnoal)
        
    #     Hess_half = self.triangular_hess_hub_sum()
    #     return Hess_half.T + Hess_half + diagnoal

    def hess_hub_pairwise(self):
        '''hess matrix of xi-xj (i<j)'''
        n, d = self.X.shape
        
        ''' hess's diagonals'''
        # lower right layer matrix
        mat1 = np.zeros((n,n*d))
        for i in range(n):
            mat1[i:, i*d:(i+1)*d] = self.X[i]                          #each column
            mat1[i, (i+1)*d:]     = np.tile(self.X[i], (n-i-1,))       #each row
        # pper left layer matrix
        mat2 = np.zeros((n,n*d))
        for i in range(n):
            mat2[:i+1, i*d:(i+1)*d] = self.X[i]                        #each column
            mat2[i, :i*d]           = np.tile(self.X[i], (i,))         #each row
        # apply hess_func to every xi-xj in (mat1-mat) 
        hess_pairwise = np.apply_along_axis(self.hess_hub, 2, (mat1-mat2).reshape(n,n,d)).reshape(n*d, n*d)
        return hess_pairwise


    def hess_product_p(self, hess_pairwise, p):
        '''Newton CG A*p_k'''
        n, d = self.X.shape
        p = p.reshape(n*d,1)
        nd = n * d
        # calculata the matrix whose diagonals are Hess's diagonals
        hess_diagonals = np.diagonal(np.dot((np.ones((nd, nd))-np.eye(nd, nd)), hess_pairwise)) # (nd,1)  arr
        
        '''hess's other elements'''
        # hess_other_ele = np.diag(np.diagonal(hess_pairwise)) - hess_pairwise                    # (nd,nd) arr
        
        '''2nd item's hess * p'''
        Ap = (hess_diagonals + np.diagonal(hess_pairwise)).reshape(-1,1) * p - np.dot(hess_pairwise, p)
        return Ap + p
        
        # == old =============================
        # Ap = []
        # for i in range(n): # each d rows of vector Hess*d
        #     hd_i = np.zeros((d,1))
        #     for k in range(n):  # sum up to calculate each d rows
        #         hd_i += np.dot(self.partial_hess_hub_sum(i, k), p[k*d : (k+1)*d])
        #     Ap.append(hd_i)
        # Ap = np.stack(Ap).reshape(-1,1)
        # return Ap + p
        

    def obj_func(self):
        '''objective function'''
        fx = 0.5*self.norm_sum_squ(a=self.a,x=self.X, squ=True) + self.lam*self.hub_sum_pairwise()
        return fx

    # def grad_obj_func0(self):
    #     '''gradient of the objective function'''

    #     grad_fx = (self.X-self.a) + self.lam*self.grad_hub_sum_pairwise()
    #     return grad_fx
    
    def grad_obj_func(self):
        '''gradient of the objective function'''

        grad_fx = (self.X-self.a) + self.lam*self.grad_hub_matrix()
        return grad_fx
    
    # def hess_obj_func(self):
    #     '''Hessian of the objective function'''
    #     first_item_hess  = np.eye(self.X.shape[0]*self.X.shape[1])
    #     second_item_hess = self.hess_hub_sum_pairwise()
    #     return first_item_hess + second_item_hess


#%% Test sample
if __name__ == "__main__":
    # a1 = self_generate_cluster(n=5, sigma=1, c = [1,1])
    # a2 = self_generate_cluster(n=5,  sigma=2, c = [3,4])
    # a  = np.concatenate((a1,a2),axis=0)
    #
    # X  = np.array([[0,0] for i in np.arange(200)])

    # X = np.array([[0, 0], [4, 3], [2, 5], [6, 6]])
    # a = np.array([[2, 2], [3, 3], [3, 3], [2, 2]])
    # f = ObjFunc(X=X, a=a, delta=1e-3, lam=1, if_use_weight=True)

    # fx = ObjFunc(X=X, a=a, delta=1e-3, lam=1)
    # print('norm: ', fx.norm_sum_squ(a))
    # print('huber:', fx.hub(X[1], X[2]))
    # print('gradient huber: ', fx.grad_hub(X[1], X[2]))
    # print('hessian huber  : ', fx.hess_hub(X[1], X[2]))

    # X  = np.array([[0,0] for i in np.arange(200)])


    # fx = ObjFunc(X = X, a = a, delta=1e-3, lam=1)
    
#%% test
    X = np.array([[0,0],[1,2],[3,5],[4,3]])
    a = np.array([[1,1],[1,1],[2,2],[2,2]])

    grad_coef = grad_hub_coef(X)
    weights   = get_weights(a, 2)
    pair_coef = pairwise_coef(X, opera = '-') 

# f = ObjFunc(X = X, a = a, delta=1e-3, lam=1, if_use_weight=True)
# f.partial_grad_hub_sum(i=1,j=1)
# grad  = f.grad_hub_sum_pairwise()
# upper = f.triangular_hess_hub_sum()
    matrix_config = {
        'gradient' : grad_coef, 
        'weights'  : weights, 
        'pairwise' : pair_coef}

    f = ObjFunc(X = X, a = a, mat_config=matrix_config,delta=1e-3, lam=1, if_use_weight=True)
    f.grad_hub_matrix()

# %%
