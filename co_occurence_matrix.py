# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:59:10 2020

@author: benja
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os
import utils
from multiprocessing import  Pool
from functools import partial
import numpy as np


#%%

    
def is_nonzero(x):
    return int(bool(x))

def coocc_matrix(df, columns, threshold = 1):
    '''Make a coocurance matrix of the columns passed.'''
    
    matrix=df[columns]

    if threshold == 1:
        matrix = matrix.apply(lambda x: (x>0).astype(int)) 
        coocc=matrix.T @ matrix
    
    else: 
        coocc = matrix.apply(lambda x: (x>=1).astype(int)).T @ matrix.apply(lambda x: (x>=threshold).astype(int))
    print(coocc)
    #coocc = coocc / coocc.sum(axis=1)
    for c in coocc.columns:
        coocc[c]=coocc[c]/matrix[c].sum() #### creating a % based matrix 
        #(% of column overlap of row )        
    return coocc.astype(float), matrix






def aut_data_retriever(name, df, columns):
    ''' '''
    sdf=df[df['AF']==name]
    #sdf=df[df['AF'].str.contains(name)]
    return sdf.loc[:, columns].sum()
    

topics=utils.load_topics() 


def make_author_df(df):
    '''Return a dictionary summarizing author data.'''
    print('Collecting Authors')
    #save 
    columns=['author_id', 'Z9']+[key for key in topics]
    print('Assembling Counts')
    counts=df['author_id'].value_counts()
    gb=df[columns].groupby('author_id')
    aut_df=gb.sum()
    del df
    print('Formatting Data')
    aut_df.rename(columns={**{'Z9': 'num_cites'}, **{key: f'num_pubs_{key}' for key in topics}}, 
                    inplace=True)
    aut_df=aut_df.merge(counts, left_index=True, right_index=True)
    
    aut_df.rename({'author_id': 'num_pubs'}, inplace=True)
    return aut_df.reset_index()
    

def make_cocc_plot(coocc, path):    
    coocc.columns=[column.replace('num_pubs_', '') for column in coocc.columns]
    coocc.index=[label.replace('num_pubs_', '') for label in coocc.index]
    coocc = coocc.replace({1:np.nan})
    vmax = coocc.max().max()
    ax = sns.heatmap(data=coocc, vmax=vmax, cmap=mpl.cm.cividis_r, linecolor='white',linewidths=1 )
    #ax.xlabels = [l.replace('num_pubs_', '') for  in coocc.columns] 
    plt.savefig(path, bbox_inches='tight')
    plt.show(block = False) 




test=False
if __name__=='__main__':
    path = os.path.join('data', 'expanded_authors.csv')
    if test:
       df=pd.read_csv(path).head(10000)
    else:
       df=pd.read_csv(path)
    topics=utils.load_topics()           
    df=utils.add_topic_cols(df)
    
    df=df.dropna(axis=0, subset=['AF', 'Z9'])
    df=df[df['Z9'].apply(utils.isnumber)]
    df['Z9']=df['Z9'].astype(int)
    for key in topics:
        df[key]=df[key].astype(int)
    
    
    
    
    
    aut_df_path=os.path.join('data', 'author_data.csv')
    if not os.path.exists(aut_df_path):
        print('Making Author DataFrame')
        author_df=make_author_df(df)  
    else:
        author_df=pd.read_csv(aut_df_path)
     
    
    
    
    print('Making Co-Occurrence Matrix')
    columns=[f'num_pubs_{topic}' for topic in topics.keys()]
    
    

    #%%
    coocc1, matrix = coocc_matrix(author_df, columns)
    make_cocc_plot(coocc1, os.path.join('figures', 'corr_pubs.png'))
    
    
    coocc2, _ = coocc_matrix(author_df, columns, 2)
    make_cocc_plot(coocc2, os.path.join('figures', 'corr_pubs2.png'))    
    
    #%%
    if not os.path.exists(aut_df_path):
        author_df.to_csv(aut_df_path)

