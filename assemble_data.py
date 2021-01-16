#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:51:44 2020
Assemble all of the data from 
@author: bdube
"""
import utils
import os
import pandas as pd
import numpy as np
import contextlib
from pandas.errors import ParserError
import re
import string

col_names=utils.load_col_names()
topics=utils.load_topics()

os.chdir(os.path.join('data', 'raw_download'))


#%%
def check_col_dtype(df, col, dtype):
    return df[col].dtype==dtype


def manual_df(fp, sep='\t'):
    '''Manually open a df from a filepath.'''
    with open(fp, 'r') as f:
        lines=[l for l in f.read().split('\n')]
        return lines_to_df(lines, split_char=sep)
       
def lines_to_df(lines, split_char='\t'):
    '''Turn a list of lists into a df.'''
    header=lines[0].split(split_char)
    header=[h.strip('\ufeff') for h in header]
    out_lines=[line.split(split_char)[:-1] for line in lines[1:] if line]
    return pd.DataFrame(out_lines, columns=header)

#%%

def check_files():
    corrupted_files=[]
    misformat_files=[]
    bad_cols_files=[]
    correct_dtypes=[np.number, np.number, np.object, np.object]
    bad_lines=[]
    bad_cols=[]
    with open('bad_lines.txt', 'w+') as log:
        with contextlib.redirect_stderr(log):
            for file in os.listdir():
                if file=='bad_lines.txt':
                    continue
                print(file, file=log)
            
                try:
                    df=pd.read_csv(file, sep='\t', index_col=False, error_bad_lines=False)
                except ParserError:
                    df=manual_df(file)
                    
                    
                try:
                    dtypes=tuple([df[col].dtype for col in ['Z9', 'PY', 'DE', 'PT']])
                except KeyError:
                    bad_cols_files.append(file)

                if not all([np.issubdtype(dtype1, dtype2) for dtype1, dtype2 in zip(dtypes, correct_dtypes)]):
                    misformat_files.append(df)
                    print(dtypes)
            
            print('\n\n', file=log)
    return bad_cols_files, misformat_files


def topic_col_maker(v, text_col):
    ['TI', 'DE', 'ID', 'AB', 'MA', 'SC']
    def specific_func(row):
        regexes=[re.compile(v_part.lower()) for v_part in v]
        return all([reg.search( str(row[text_col]).lower()) for reg in regexes ])
                
        
    return specific_func
    
    



#%%  
def add_TopicColumns(df, topics):
    '''Add topic columns to df'''
    for k, v in topics.items():
        function=topic_col_maker(v)
        df[k]=df.apply(function, axis=1)
    return df

def make_combined_text_col(df, columns):
    'Aggregate text from df[columns into a single series.'
    sub=df[columns].astype(str)
    return sub[columns[0]].str.cat(sub[columns[1:]].astype(str), sep=' ')


#%%
def topic_cols(df, topics, cols_to_search):
    '''Make boolean columns for topics. For each column, represents whether
    any of the cols_to_search match to that column.
    Args: Dataframe.
    topics: a dictionary where key is the name of the column to be made,
    value is list of regexes to match.
    cols_to_search: list of column names to search.
    Returns: dataframe.'''
    
    df['all_text']=make_combined_text_col(df, cols_to_search)
    for key,values in topics.items():
        df[key]=np.logical_and(*[df['all_text'].str.contains(v) for v in values])
    return df.drop(columns='all_text')
#%%
def load_data():
    dfs=[]
    bad_dfs= []
    for file in os.listdir():
        if file=='bad_lines.txt':
            continue
    
        try:
            df=pd.read_csv(file, sep='\t', index_col=False, error_bad_lines=False)
        except ParserError:
            print('Parser Error')
            print(file)
            df=manual_df(file)
        assert tuple(df.columns)==('PT', 'AU', 'BA', 'BE', 'GP', 'AF', 'BF', 'CA', 'TI', 'SO', 'SE', 'BS',
           'LA', 'DT', 'CT', 'CY', 'CL', 'SP', 'HO', 'DE', 'ID', 'AB', 'C1', 'RP',
           'EM', 'RI', 'OI', 'FU', 'FX', 'CR', 'NR', 'TC', 'Z9', 'U1', 'U2', 'PU',
           'PI', 'PA', 'SN', 'EI', 'BN', 'J9', 'JI', 'PD', 'PY', 'VL', 'IS', 'PN',
           'SU', 'SI', 'MA', 'BP', 'EP', 'AR', 'DI', 'D2', 'EA', 'PG', 'WC', 'SC',
           'GA', 'UT', 'PM', 'OA', 'HC', 'HP', 'DA')
        if check_df(df):
            dfs.append(df)
        else:
            df= reload_df(file)    
            assert check_df(df)
            dfs.append(df)
    return pd.concat(dfs).reset_index(), bad_dfs

#%%
def check_df(df):
    '''Run three different checks that df has proper formatting'''
    return df_checker(df) and df_checker2(df) and df_checker3(df)


def df_checker(df):
    if df['EI'].str.contains('|'.join(string.ascii_letters.replace('X', ''))).sum() >0:
        return False
    return True

def reload_df(fp):
    '''Fix misformatted df by reading the text file, re-writing it with all 
    quotation marks taken out, and reading it back in. '''
    
    with open(fp, 'r') as file:
        text = file.read()
    text = text.replace("'", '').replace('"', '')
    with open('temp.txt', 'w+') as file2:
        print(text, file = file2)
    
    df = pd.read_csv('temp.txt', sep='\t', 
                       index_col=False, error_bad_lines=False)
    os.remove('temp.txt')
    return df
    

def df_checker2(df):
    return df['TI'].str.contains("[\'\"]").sum() == 0

def df_checker3(df):
    return df['PY'].apply(lambda x: type(x) not in [float, int]).sum() == 0

EI_re = re.compile('|'.join(string.ascii_letters.replace('X', '')))
def ID_num_checker(n):
     if EI_re.search(n):
         return False
     return True




if __name__=='__main__':
    
    df, bad_dfs=load_data()
    
    print('Formatting Data')
    cols_to_search=['TI', 'DE', 'ID', 'AB', 'MA', 'SC', 'CT', 'SE', 'BS']
    df=topic_cols(df, topics, cols_to_search)
    
    df['any_topic']=df[list(topics.keys())].sum(axis=1)
    df['ID_num'] = None
    
    df.loc[df['DI'].isnull() ==False, 'ID_num'] = df['DI'].dropna()
    df.loc[
        (df['DI'].isnull()) & (df['SN'].isnull() == False), 'ID_num'
        ] = 'book chap ' + df['SN'] + ' // ' +df.index.astype(str)
    
    
    df.loc[df['ID_num'].isnull(), 'ID_num'] = df[df['ID_num'].isnull()]['UT']
    print('Saving Data')
    with utils.cwd('..'):
        df.to_csv('all_data.csv')       
    