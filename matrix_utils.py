#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 15:51:31 2021

@author: bdube
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import utils
import numpy as np
from utils import topics

#%%
'''This script makes an author overlap matrix.'''
    
def is_nonzero(x):
    return int(bool(x))

def coocc_matrix(df, columns, threshold = 1):
    '''Make a coocurance matrix of the columns passed.'''
    
    matrix=df[columns]

    if threshold == 1:
        matrix = matrix.apply(lambda x: (x>0).astype(int)) 
        coocc=matrix.T @ matrix
        for c in coocc.columns:
            coocc[c]=coocc[c]/matrix[c].sum()
    else: 
        matrix1 = matrix.apply(lambda x: (x>1).astype(int)).T
        matrix2 = matrix.apply(lambda x: (x>=threshold).astype(int))
        coocc =  matrix1 @ matrix2
    
    #coocc = coocc / coocc.sum(axis=1)
        for c in coocc.columns:
            coocc[c]=coocc[c]/matrix2[c].sum() #### creating a % based matrix 
        #(% of column overlap of row )        
    return coocc.astype(float), matrix


def weighted_cocc_matrix(df, columns):
    '''Make a coocurrence matrix  '''
    matrix = df[columns]
    coocc = matrix.astype(bool).astype(int).T @ matrix
    for c in coocc.columns:
        coocc[c]=coocc[c]/matrix[c].sum()
 
    return coocc.astype(float), matrix



def aut_data_retriever(name, df, columns):
    '''Aggregate all data for an author of name: name'''
    sdf=df[df['AF']==name]
    #sdf=df[df['AF'].str.contains(name)]
    return sdf.loc[:, columns].sum()
    

def nan_identity(n):
    '''Make an n-by-n matrix with nan values on the diagonal.'''
    a = (np.identity(12)-1)*-1
    return np.where(a==0, np.nan, a)

def check_keep(row):
    '''Return true if none of the columns in source equal their corresponding
    column in target.'''
    return ((utils.filter_series(row, 
            'num_pubs_[\w_\s]+_source').to_numpy()*utils.filter_series(
                row, 'num_pubs_[\w_\s]+_target').to_numpy())==0).all()

                
def cocc_plot(coocc, path):    
    '''Make a heatmap plot from a coocc matrix and save it to path.'''
    coocc = rm_str_from_cols(coocc, 'num_pubs_')
    coocc.index=list(coocc.columns)
    coocc = coocc.replace({1:np.nan})
    vmax = coocc.max().max()
    ax = sns.heatmap(data=coocc, vmax=vmax, cmap=mpl.cm.cividis_r, linecolor='white',linewidths=1 )
    #ax.xlabels = [l.replace('num_pubs_', '') for  in coocc.columns] 
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    #plt.show(block = False) 
    


    

def rm_str_from_cols(df, string):
    '''Remove a string from all column names where it appears in a df.'''
    return df.rename(columns = {c: c.replace(string, '') for c in df.columns})

def make_coocc_autcollab(edges, topics, col_string = 'num_pubs_',
                         directed = False):
    '''Make a co_occurrence matrix based on two-sided data for author collaborations.
    args_ : 
        _edges_ : a df with 2-sided data. should have columns: source and target
        and [{name}_source for name in topics] and [{name}_target for name in topics]
        topics : names of data columns
        directed: bool, whether the network is directed or not. 
        If not directed, all (source, target) pairs are also analyzed as
        (target, source) pairs.
    '''
    
    out_data = [] 
    for topic in topics:
        sub1 = edges[edges[f'{col_string}{topic}_source']]
        sub2 = edges[edges[f'{col_string}{topic}_target']]
        sub1 = sub1[utils.parallelize_on_rows(sub1, check_keep)]
        sub2 = sub2[utils.parallelize_on_rows(sub2, check_keep)]
        if directed:
            data = rm_str_from_cols(utils.filter_cols(sub1, 
        f'{col_string}[\w_\s]+_target'), '_target')
            
        else:
            data = pd.concat(
                    [rm_str_from_cols(
                        utils.filter_cols(sub, 
                            f'{col_string}[\w_\s]+{s}'), 
                        s)
             for sub, s in [(sub1, '_target'), (sub2, '_source')]
             ]
            )
        data = rm_str_from_cols(data, col_string)
        out_data.append(data.mean())
    return out_data


def only_topic_cols(df, topics=topics):
    '''Return a list of column names which contains one of the strings in topics.'''
    return [c for c in df.columns if any([t in c for t in topics])]

def network_coocc(edges, columns):
    '''Make a cooccurence matrix from network data.  
    '''
    edges_agg = edges.groupby('source')[only_topic_cols(edges, columns)].sum()
    edges_agg = edges_agg.astype(bool)
    a1 = np.array(edges_agg[[f'{column}_source' for column in columns]]).astype(int).T
    a2 = np.array(edges_agg[[f'{column}_target' for column in columns]]).astype(int)
    coocc = a2.T @ a1.T
    new_coocc = np.zeros((12,12))
    for i in range(coocc.shape[0]):
        new_coocc[:,i] = coocc[:,i]/edges_agg[f'{columns[i]}_source'].sum()
    return pd.DataFrame(new_coocc, index = columns, columns = columns)
