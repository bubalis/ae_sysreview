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

def coocc_matrix(df, columns):
    '''Make a coocurance matrix of the columns passed.'''
    
    matrix=df[columns]
    for column in matrix.columns:
        matrix[column]=matrix[column].apply(is_nonzero)
    coocc=matrix.T.dot(matrix)
    print(coocc)
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




def split_rows_df(df):
    return pd.DataFrame(utils.list_flattener(df.apply(split_row, axis=1).tolist()))

def make_author_df(a_list, df):
    '''Return a dictionary summarizing author data.'''
    print('Collecting Authors')
    all_aut_df=utils.split_rows_paralell(df)
    aut_df=pd.DataFrame(data=a_list, columns=['Author Name'])
    #save 
    columns=['AF', 'Z9']+[key for key in topics]
    print('Assembling Counts')
    counts=all_aut_df['AF'].value_counts()
    gb=all_aut_df[columns].groupby('AF')
    add_data=gb.sum()                   
    print('Formatting Data')
    add_data.rename(columns={**{'Z9': 'num_cites'}, **{key: f'num_pubs_{key}' for key in topics}}, 
                    inplace=True)
    aut_df=aut_df.merge(counts, left_on='Author Name', right_index=True)
    aut_df.rename({'AF': 'num_pubs'}, inplace=True)
    return aut_df.merge(add_data, 
                        left_on='Author Name', right_index=True)
    
#%%

    
def make_author_df_old(a_list, df):
    '''Slow way to make author dataframe.'''
    aut_dict={}
    for auth in a_list:
        aut_dict[auth]={}
        #sdf=df[df['AF'].str.contains(auth)]
        sdf=df[df['AF']==auth]
        aut_dict[auth]['num_pubs']=sdf['AF'].count()
        aut_dict[auth]['num_cites']=sdf['Z9'].sum()
        for key in topics: 
            aut_dict[auth]['num_pubs_{}'.format(key)]=sdf[key].sum()
    return pd.DataFrame(aut_dict).transpose()

#%%
df=pd.read_csv(os.path.join("data","corrected_authors.csv"))
df=df.drop(columns=[col for col in df.columns if 'Unnamed' in col])
with open(os.path.join('data', 'author_list.txt')) as f:
        a_list=[author for author in f.read().split('\n')]
#%%     
test_set=a_list[:2000]

#%%




test=False
if __name__=='__main__':
    if test:
       df=pd.read_csv(os.path.join("data","corrected_authors.csv")).head(10000)
    else:
       df=pd.read_csv(os.path.join("data","corrected_authors.csv"))
    topics=utils.load_topics()           
    
    
    
    
    df=df.dropna(axis=0, subset=['AF', 'Z9'])
    df=df[df['Z9'].apply(utils.isnumber)]
    df['Z9']=df['Z9'].astype(int)
    for key in topics:
        df[key]=df[key].astype(int)
    
    with open(os.path.join('data', 'author_list.txt')) as f:
        a_list=[author for author in f.read().split('\n')]
    
    
    
    aut_df_path=os.path.join('data', 'author_data.csv')
    if not os.path.exists(aut_df_path):
        print('Making Author DataFrame')
        author_df=make_author_df(a_list, df)  
    else:
        author_df=pd.read_csv(aut_df_path)
     
    
    
    
    print('Making Co-Occurrence Matrix')
    columns=[f'num_pubs_{topic}' for topic in topics.keys()]
    
    
    
    coocc, matrix=coocc_matrix(author_df, columns) #% of authors who publish in Column who also publish in Row
    #%%
    coocc.columns=[column.replace('num_pubs ', '') for column in coocc.columns]
    coocc.index=[label.replace('num_pubs ', '') for label in coocc.index]
    
    coocc=coocc.replace({1:np.nan})
    unique_values=utils.list_flattener(coocc.values.tolist())
    
    vmax=max([v for v in unique_values if not np.isnan(v)])
    sns.heatmap(data=coocc, vmax=vmax, cmap=mpl.cm.cividis_r, linecolor='white',linewidths=1 )
    fig_path=os.path.join('figures', 'corr_pubs.png')
    plt.savefig(fig_path, bbox_inches='tight')
    #%%
    if not os.path.exists(aut_df_path):
        author_df.to_csv(aut_df_path)

