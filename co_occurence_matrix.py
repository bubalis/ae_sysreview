# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:59:10 2020

@author: benja
"""

import pandas as pd
import os
import utils
import matrix_utils

#%%
'''This script makes an author overlap matrix.'''
    
topics=utils.load_topics()

def aut_data_retriever(name, df, columns):
    ''' '''
    sdf=df[df['AF']==name]
    #sdf=df[df['AF'].str.contains(name)]
    return sdf.loc[:, columns].sum()
    

 
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
    author_name_data = pd.read_csv(os.path.join('data', 
                                                'intermed_data', 'all_author_data.csv'))
    author_name_data = author_name_data.drop_duplicates(subset = ['author_id'])
    aut_df = aut_df.merge(author_name_data[['author_id', 'longer_name']],
                          left_index = True, right_on = 'author_id')
    aut_df.drop(columns = ['author_id_y','author_id_x'], inplace = True)
    return aut_df.reset_index()
    





test=False
if __name__=='__main__':
    path = os.path.join('data', 'intermed_data', 'expanded_authors.csv')
    if test:
       df=pd.read_csv(path, nrows = 50000)
    else:
       df=pd.read_csv(path)
    topics=utils.load_topics()           
    df=utils.add_topic_cols(df)
    
    df=df.dropna(axis=0, subset=['AF', 'Z9'])
    df=df[df['Z9'].apply(utils.isnumber)]
    df['Z9']=df['Z9'].astype(int)
    for key in topics:
        df[key]=df[key].astype(int)
    
    
    
    
    
    aut_df_path=os.path.join('data','intermed_data', 'author_pub_data.csv')
    if not os.path.exists(aut_df_path):
        print('Making Author DataFrame')
        author_df=make_author_df(df)  
    else:
        author_df=pd.read_csv(aut_df_path)
     
    
    
    
    print('Making Co-Occurrence Matrix')
    columns=[f'num_pubs_{topic}' for topic in topics.keys()]
    
    

    #%%
    coocc1, matrix = matrix_utils.weighted_cocc_matrix(author_df, columns)
    matrix_utils.cocc_plot(coocc1, os.path.join('figures', 'corr_pubs_weighted.png'))
    coocc1.to_csv(os.path.join('data', 'matrixes', 'coor_pubs_weighted.csv'))
    coocc2, _ = matrix_utils.coocc_matrix(author_df, columns)
    matrix_utils.cocc_plot(coocc2, os.path.join('figures', 'corr_pubs_author.png'))    
    coocc2.to_csv(os.path.join('data', 'matrixes', 'coor_pubs_author.csv'))
    
    
    #%%
    if not os.path.exists(aut_df_path):
        author_df.to_csv(aut_df_path)

