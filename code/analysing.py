#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   analysing.py
@Time    :   2021/12/28 22:43:13
@Author  :   Lin Junwei
@Version :   1.0
@Desc    :   None
'''

#%% 
from initialization import load_dataset, self_generate_cluster, grad_hub_coef, self_dataset, get_weights, pairwise_coef
from initialization import ObjFunc
from initialization import log_read, pickle_read, pickle_write
import numpy as np
import pandas as pd
import time
import logging
import pickle
import os
import matplotlib.pyplot as plt
#%% 
def cluster_check(X, max = 20):
    K = range(1, max)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])
        plt.plot(K, meandistortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Average Dispersion')
        plt.title('Selecting k Centroids')
        plt.show()

def simpleToPlot(arr, pic_path = str(os.getcwd())+'\\pic'):

    pd.DataFrame(arr).plot.scatter(x=0, y=1, c=c)
    plt.savefig(pic_path+'\\a.png')
    plt.show()

def picklesToPlot(label, folder = 'AGM1', L_R_U_D=[None, None, None, None] , max_pic=50, pic_folder = None, show_pic = False):
    '''
    Plot pickle
    only feasible in 2D 

    label      : label array of the scatter data
    L_R_U_D    : LEFT, RIGHT, UPPER, DOWN limit in list 
    folder     : folder in the result, default 'AGM1'
    max_pic    : how many iterations to be plotted, default 50
    pic_folder : if None then use the 'folder' name
    show_pic   : True it will print on console

    '''

    if pic_folder is not None:
        pass
    else:
        pic_folder = folder
    
    pic_path = str(os.getcwd())+ '\\pic\\' + pic_folder
    if not os.path.exists(pic_path):    
        os.makedirs(pic_path)
        print("--- Pictures will be saved in : ", pic_path, ' --- ')

    ### color dict ###
    def generate_color(kind):
        color_ls = ['r','g','b','#1e1e1e','y']
        for i in range(kind):
            color_ls.extend(["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])])
        return color_ls 

    categories = list(set(label.reshape(-1,).tolist()))
    colors_ls  = generate_color(len(categories))[:len(categories)]
    # color - categories mapping
    color_dict = dict(zip(categories, colors_ls))
    
    c = pd.Series(label.reshape(-1,)).apply(lambda x: color_dict[x])

    #### pickle read ####
    filels = os.listdir(str(os.getcwd()) + '\\result\\' + folder)
    array_ls = []

    for file in filels[:max_pic]:
        file = file.split('.')[0]
        df = pickle_read(file, folder = folder) # read pickles
        array_ls.append(df)
        # Plotting 
        pd.DataFrame(arr).plot.scatter(x=0, y=1, c=c)
        plt.xlim(L_R_U_D[0], L_R_U_D[1])
        plt.ylim(L_R_U_D[2], L_R_U_D[3])

        plt.savefig(pic_path+'\\'+file+'.png')
        if not show_pic:
            print(file, 'th-----------------------')
            plt.show()
    
    return array_ls
#%% gradient convergence plot
def convergence_plot(logname, title_nm = None, max_iter = 100):
    '''sinlge convergence plot'''
    df = log_read(logname=logname)
    # logname = 'weighted_AGM_delta1e-1_lam1e-1_tol1_wine(1)'
    df['grad'] = df['grad'].apply(lambda x: round(float(x),4))
    df['iter'] = df['iter'].apply(lambda x: int(x))
    df = df.set_index('iter')['grad'][:max_iter]

    plt.figure(figsize=(10, 6))
    plt.plot(df)
    plt.xlabel('Iteration', fontsize=16)
    if title_nm is None:
        pass
    else:
        plt.title(title_nm)
    plt.ylabel(r'$\nabla f(X^{k})$', fontsize=20)
    # plt.legend(fontsize = 16)
    plt.savefig(os.getcwd()+'\\pic\\convergence\\'+ logname +'_convergence.png',bbox_inches='tight')
    print('saved in ', os.getcwd()+'\\pic\\convergence\\'+ logname +'_convergence.png')
    plt.show()

def multi_convergence(log_ls, fig_name,title_nm = None, max_iter = 100, legend_name = None):
    '''
    Compare difference convergence in 1 fig
    
    legend_name - list of the dataframe columns name, default filename  

    '''
    
    df_ls = []
    for lg in log_ls:
        # logname = 'weighted_AGM_delta1e-1_lam1e-1_tol1_wine(1)'
    
        df = log_read(logname=logname)
        df['grad'] = df['grad'].apply(lambda x: round(float(x),4))
        df['iter'] = df['iter'].apply(lambda x: int(x))
        df = df.set_index('iter')['grad']
        df_ls.append(df)
    df = pd.concat(df_ls, axis=1)[:max_iter]
    if legend_name is None:
        df.columns = log_ls
    else:
        df.columns = legend_name

    plt.figure(figsize=(10, 6))
    df.plot()
    plt.xlabel('Iteration', fontsize=16)
    if title_nm is None:
        pass
    else:
        plt.title(title_nm)
    plt.ylabel(r'$\nabla f(X^{k})$', fontsize=20)
    plt.savefig(os.getcwd()+'\\pic\\convergence\\'+ fig_name +'_convergence.png',bbox_inches='tight')
    print('saved in ', os.getcwd()+'\\pic\\convergence\\'+ fig_name +'_convergence.png')
    plt.show()
