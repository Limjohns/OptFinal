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
import imageio
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import random
#%% 

def cluster_norm(pickle_nm = None, folder='AGM1', tol = 0.5):
    ''' 
    Cluster check by norms and dfs searching
    input - pickle name, str or int

    Return
    ----------
    cluster label array : array with length n, labelled with 1,2,3....
    '''
    # pickle_nm = '9'
    # folder='AGM1'
    # tol = 0.2
    if pickle_nm is None: # return the last iteration
        pickle_nm = max([int(pick.split('.')[0] )for pick in os.listdir(os.getcwd()+'\\result\\'+folder)])
    
    X = pickle_read(str(pickle_nm),folder=folder)
    norm_pair = distance_matrix(X,X)


    n = norm_pair.shape[0]

    def dfs_neighbor(idx=0):
        stack, dfs_path = [idx], []

        while stack:
            vertex = stack.pop()
            if vertex in dfs_path:
                continue
            dfs_path.append(vertex)
            neighbor_ls = np.array([j for j in range(n)])[(norm_pair[idx] < tol)]
            
            for neighbor in neighbor_ls:
                stack.append(neighbor)
                stack = list(set(stack))
        return dfs_path
    
    start       = 0
    cluster_idx = dfs_neighbor(idx=start) # 0 point's cluster members
    rest_idx    = [i for i in range(n) if i not in cluster_idx]

    cluster_ls  = [cluster_idx]

    while rest_idx:
        second   = dfs_neighbor(idx=rest_idx[0])
        cluster_ls.append(second)
        rest_idx = [i for i in rest_idx if i not in second]

    clus_num  = len(cluster_ls)

    print('tol =', tol ,' -  Total clusters: ',clus_num)
    
    cluster_label_arr = np.array([i for i in range(n)])

    for i in range(0, clus_num):
        cluster_label_arr[cluster_ls[i]] = i +1


    return cluster_label_arr


#%% 
def plotsToGiF(pic_path, gifname):
    '''convert a folder pictures to gif'''
    images = []
    for filename in pic_path:
        images.append(imageio.imread(filename))
    imageio.mimsave('/gif'+ gifname +'.gif', images)

def simpleToPlot(arr, pic_path = str(os.getcwd())+'\\pic'):

    pd.DataFrame(arr).plot.scatter(x=0, y=1, c=c)
    plt.savefig(pic_path+'\\a.png')
    plt.show()


def rawToPlot(a, folder = 'AGM1', tol = 0.5, L_R_U_D=[None, None, None, None] , max_pic=50, pic_folder = None, show_pic = False):
    '''
    a      : row data 
    
    folder : pickles folder  

    show how categories changes with iteration 
    
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

    

    #### pickle read ####
    filels = os.listdir(str(os.getcwd()) + '\\result\\' + folder)

    for file in filels[:max_pic]:
        file  = file.split('.')[0]

        label = cluster_norm(pickle_nm = file, folder=folder, tol = tol)

        c     = pd.Series(label.reshape(-1,)).apply(lambda x: color_dict[x])

        categories = list(set(label.reshape(-1,).tolist()))
        colors_ls  = generate_color(len(categories))[:len(categories)]
        # color - categories mapping
        color_dict = dict(zip(categories, colors_ls))
        # Plotting 
        pd.DataFrame(a).plot.scatter(x=0, y=1, c=c)
        plt.xlim(L_R_U_D[0], L_R_U_D[1])
        plt.ylim(L_R_U_D[3], L_R_U_D[2])
        plt.title(file + 'th Iteration')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(pic_path+'\\'+file+'.png')
        if not show_pic:
            print(file, 'th-----------------------')
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

    for file in filels[:max_pic]:
        file = file.split('.')[0]
        df = pickle_read(file, folder = folder) # read pickles
        # Plotting 
        pd.DataFrame(df).plot.scatter(x=0, y=1, c=c)
        plt.xlim(L_R_U_D[0], L_R_U_D[1])
        plt.ylim(L_R_U_D[3], L_R_U_D[2])
        plt.title(file + 'th Iteration')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(pic_path+'\\'+file+'.png')
        if not show_pic:
            print(file, 'th-----------------------')
            plt.show()
    




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
        plt.title(title_nm, fontsize=20)
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
    
        df = log_read(logname=lg)
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
        plt.title(title_nm, fontsize=16)
    plt.ylabel(r'$\nabla f(X^{k})$', fontsize=20)

    if legend_name is not None:
        plt.legend()
    
    plt.savefig(os.getcwd()+'\\pic\\convergence\\'+ fig_name +'_convergence.png',bbox_inches='tight')
    print('saved in ', os.getcwd()+'\\pic\\convergence\\'+ fig_name +'_convergence.png')
    plt.show()
#%% run time plot

fl_ls     = os.listdir(os.getcwd()+'\\log')
fl_ls     = ['E:\\OneDrive\\2021-2022 研一上\\MDS6106-Optimization\\homework\\Final\\log\\bck_NCG_delta0.1_lam0.1_tol1_wine.log']
log_ls    = []
legend_ls = []
df_ls  = []

for file in fl_ls:
    file  = file.split('.log')[0]
    df    = log_read(file)

    title = file.split('_')
    # title = 'Backtrack NCG'+'-'+ title[3] 
    # convergence_plot(file, title_nm = title, max_iter = None)
    legend_ls.append(title[3])
    log_ls.append(file)
    df_ls.append(df)
    

    print(title[3])
    print(df['time'].apply(lambda x: float(x)).sum(),'s')
    cluster_norm(pickle_nm = None, folder=file, tol = 0.05)
    print('---------------------')


# title_nm = 'Backtrack NCG'
# multi_convergence(log_ls, fig_name = 'bkt_ncg_diff_lambda_',title_nm = title_nm, max_iter = 100, legend_name = legend_ls)




# %%
