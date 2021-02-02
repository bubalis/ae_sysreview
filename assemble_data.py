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
from pandas.errors import ParserError, EmptyDataError
import re
import string

col_names=utils.load_col_names()
topics=utils.load_topics()



#%%
def float_maker(d):
    if not d:
        return np.nan
    else:
        return float(d)

def check_col_dtype(df, col, dtype):
    return df[col].dtype==dtype


def manual_df(fp, sep='\t'):
    '''Manually open a df from a filepath.'''
    with open(fp, 'r') as f:
        lines=[l for l in f.read().split('\n')]
        df = lines_to_df(lines, split_char=sep)
        df['PY'] = df['PY'].apply(float_maker)
        df['Z9'] = df['Z9'].apply(float_maker)
        return df
       
def lines_to_df(lines, split_char='\t'):
    '''Turn a list of lists into a df.'''
    header=lines[0].split(split_char)
    header=[h.strip('\ufeff') for h in header]
    out_lines=[line.split(split_char)[:-1] for line in lines[1:] if line]
    return pd.DataFrame(out_lines, columns=header)

#%%
correct_dtypes=[np.number, np.number, np.object]
def check_df_dtypes(df):
    try:
        dtypes=tuple([df[col].dtype for col in ['Z9', 'PY', 'DE']])
    except KeyError:
        return False
    
    if not all([np.issubdtype(dtype1, dtype2) for dtype1, dtype2 in zip(dtypes, correct_dtypes)]):
        if ((df['DE'].isnull() == False).sum()) == 0 and (
        all ([np.issubdtype(dtype1, dtype2) for dtype1, dtype2 in zip(dtypes[:-1], correct_dtypes[:-1])])):
            return 
        else:
            return False
    return True



def check_files():
    corrupted_files=[]
    misformat_files=[]
    bad_cols_files=[]
   
    bad_lines=[]
    bad_cols=[]
    with open('bad_lines.txt', 'w+') as log:
        #with contextlib.redirect_stderr(log):
        for file in os.listdir():
            if file=='bad_lines.txt':
                continue
            print(file)
        
            try:
                df=pd.read_csv(file, sep='\t', index_col=False, error_bad_lines=False)
            except ParserError:
                df=manual_df(file)
            
                
                
            
        
        print('\n\n')
    return bad_cols_files, misformat_files


#%%
def load_data(directory):
    with utils.cwd(directory):
        dfs=[]
        bad_dfs= []
        for file in os.listdir():
            #print(file)
            if file=='bad_lines.txt':
                continue
        
            try:
                df=pd.read_csv(file, sep='\t', index_col=False, 
                               #error_bad_lines=False
                               )
            except ParserError:
                print('Parser Error')
                #print(file)
                df=manual_df(file)
                if df.empty:
                    bad_dfs.append(file)
            except EmptyDataError:
                bad_dfs.append(file)
                continue
                
            if tuple(df.columns)!=('PT', 'AU', 'BA', 'BE', 'GP', 'AF', 'BF', 'CA', 'TI', 'SO', 'SE', 'BS',
               'LA', 'DT', 'CT', 'CY', 'CL', 'SP', 'HO', 'DE', 'ID', 'AB', 'C1', 'RP',
               'EM', 'RI', 'OI', 'FU', 'FX', 'CR', 'NR', 'TC', 'Z9', 'U1', 'U2', 'PU',
               'PI', 'PA', 'SN', 'EI', 'BN', 'J9', 'JI', 'PD', 'PY', 'VL', 'IS', 'PN',
               'SU', 'SI', 'MA', 'BP', 'EP', 'AR', 'DI', 'D2', 'EA', 'PG', 'WC', 'SC',
               'GA', 'UT', 'PM', 'OA', 'HC', 'HP', 'DA'):
                bad_dfs.append(file)
                continue
            
            if not check_df_dtypes(df):
                df= reload_df(file)   
                if not check_df_dtypes(df):
                    bad_dfs.append(df)
                    continue
                
            
            try:
                if check_df(df):
                    pass
                else:
                    print('Reloading due to bad formatting')
                    df= reload_df(file)    
                    assert check_df(df)
            except:
                print('Reloading')
                df = manual_df(file)
                try:
                    assert check_df(df)
                except:
                    bad_dfs.append(file)
                    continue
            dfs.append(df)
    return pd.concat(dfs).reset_index(), bad_dfs

#%%
def check_df(df):
    '''Run three different checks that df has proper formatting'''
    return df_checker(df) and df_checker2(df) and df_checker3(df)


def df_checker(df):
    if (df['EI'].isnull() ==False).sum()>0:
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
    
    df = manual_df('temp.txt', sep='\t')
                      
    os.remove('temp.txt')
    return df
    
def try_float(d):
    if not d:
        return True
    try:
        float(d)
        return True
    except:
        return False


def df_checker2(df):
    return df['TI'].str.contains("[\'\"]").sum() == 0

def df_checker3(df):
    if df['PY'].apply(lambda x: type(x) not in [float, int]).sum() == 0:
        return True
    else:
        return (df['PY'].apply(try_float)==False).sum() == 0
    
EI_re = re.compile('|'.join(string.ascii_letters.replace('X', '')))
def ID_num_checker(n):
     if EI_re.search(n):
         return False
     return True



if __name__=='__main__':
    directory = os.path.join('data', 'raw_download2')
    df, bad_dfs=load_data(directory)
    assert not bad_dfs
    print(df.shape)
    
    print('Formatting Data')
    df.loc[df['DI'].astype(bool)==False, 'DI'] = np.nan
    
    
    print('Saving Data')
    df =df.drop_duplicates(subset=['AU', 'AF', "TI", 'DI']).reset_index()
    print(df.shape)
    save_path = os.path.join('data', 'all_data.csv')
    df.to_csv(save_path)    
    