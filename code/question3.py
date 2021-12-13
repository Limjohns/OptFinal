# -*- coding: utf-8 -*-
'''
@File    :   question3.py
@Time    :   2021/12/13 15:05:37
@Author  :   Zheng Mengchu
@Version :   1.0
@Desc    :   processing file for question3
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
