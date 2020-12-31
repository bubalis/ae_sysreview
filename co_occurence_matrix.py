# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:59:10 2020

@author: benja
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import utils
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





    
    
def make_author_df(a_list, df):
    '''Return a dictionary summarizing author data.'''
    
    
    aut_dict={}
    for auth in a_list:
        aut_dict[auth]={}
        sdf=df[df['AF'].str.contains(auth)]
        aut_dict[auth]['num_pubs']=sdf['AF'].count()
        aut_dict[auth]['num_cites']=sdf['Z9'].sum()
        for key in topics: 
            aut_dict[auth]['num_pubs_{}'.format(key)]=sdf[key].sum()
    return pd.DataFrame(aut_dict).transpose()



#%%
if __name__=='__main__':
    topics=utils.load_topics()           
    df=pd.read_csv(os.path.join("data","corrected_authors.csv"))
    
    
    
    df=df.dropna(axis=0, subset=['AF', 'Z9'])
    df=df[df['Z9'].apply(utils.isnumber)]
    df['Z9']=df['Z9'].astype(int)
    for key in topics:
        df[key]=df[key].astype(int)
    
    with open(os.path.join('data', 'author_list.txt')) as f:
        a_list=[author for author in f.read().split('\n')]
    
    
    author_df=make_author_df(a_list, df)  
     
    
    author_df.sort_values(by='num_cites',ascending=False).head(20)
    #%%
    columns=[f'num_pubs_{topic}' for topic in topics.keys()]
    
    
    
    coocc, matrix=coocc_matrix(author_df, columns) #% of authors who publish in Column who also publish in Row
    #%%
    coocc.columns=[column.replace('num_pubs ', '') for column in coocc.columns]
    coocc.index=[label.replace('num_pubs ', '') for label in coocc.index]
    
    
    unique_values=utils.list_flattener(coocc.values.tolist())
    vmax=max([u for u in unique_values if round(u, 5)!=1])*1.1
    
    sns.heatmap(data=coocc, vmax=vmax, cmap='coolwarm', linecolor='white',linewidths=1 )
    fig_path=os.path.join('figures', 'corr_pubs.png')
    plt.savefig(fig_path, bbox_inches='tight')
    #%%
    save_path=os.path.join('data', 'author_data.csv')
    author_df.to_csv(save_path)

