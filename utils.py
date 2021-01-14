#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:17:21 2020

@author: bdube
"""
import json
import os
from multiprocessing import  Pool
from functools import partial
import pandas as pd
import numpy as np

def parallelize(data, func, num_of_processes=8):
    '''Function for paralellizing any function on a dataframe.
    Stolen from stack overflow, user Tom Raz:
    https://stackoverflow.com/questions/26784164/pandas-multiprocessing-apply'''
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data



def run_on_subset(func, data_subset):
    '''For use in parallelize.
    Stolen from stack overflow, user Tom Raz:
    https://stackoverflow.com/questions/26784164/pandas-multiprocessing-apply'''
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=8):
    '''Apply a function to every row in a pandas df in parallell.
    Stolen from stack overflow, user Tom Raz:
    https://stackoverflow.com/questions/26784164/pandas-multiprocessing-apply''' 
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def split_row(row, split_col='AF', sep=';'):
    '''Split a row into multiple rows based on a string list in the split_col.
    Returns: a list of dictionaries'''
    return [{**{col:row[col] for col in row.index },**{split_col:name.strip()}} 
            for name in row[split_col].split(sep)]


def split_row_multi(row, split_cols=['AF', 'AU'], seps=[';', ';']):
    '''Split rows in df, preserving all other features, 
    by exploding multiple string-encoded lists into their own rows.'''
    
    splits = [row[col].split(sep) for col, sep in zip(split_cols, seps)]
    out = []
    
    for tup in zip(*splits):
        out.append({**{col:row[col] for col in row.index},
                    **{split_col:name.strip() for name, split_col 
                       in zip(tup, split_cols)}}) 
    return out

def split_rows_paralell(df, split_col='AF', sep=';'):
    '''Paralellized function for exploding a df by creating 
    a new row for each item in a string list contained in a column of the df.
    args: df- dataframe to be split.
    split_col: the column that contains a list as string.
    sep: character that splits the list. 
    Returns: a df where each item in row[split_col] is returned as its own row,
    with data in all other columns copied.'''
    
    func= lambda row: split_row(row, split_col, sep)
    return pd.DataFrame(list_flattener(
         parallelize_on_rows(df, func).tolist()))


from contextlib import contextmanager


this_dir=os.path.split(os.path.realpath(__file__))[0]
def load_topics():
    return json.loads(
        open(os.path.join(this_dir, 'data', 'keys', 'topic_dic.txt')).read())


def load_col_names():
    return json.loads(
        open(os.path.join(this_dir, 'data', 'keys', 'column_names.txt')).read())
#%%
@contextmanager
def cwd(path):
    orig_path=os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)

@contextmanager
def progress_saver(*filepaths, ds_types=[dict, dict]):
    data_sets=[]
    for fp, ds_type in zip(filepaths, ds_types):
        if os.path.exists(fp):
            data_sets.append(json.loads(open(fp).read()))
        else:
            data_sets.append(ds_type())
    print(data_sets)
    try:
        yield data_sets
    finally:
        for fp, ds in zip(filepaths, data_sets):
            with open(fp, 'w+') as f:
                print(json.dumps(ds), file=f)

def isnumber(string):
    try:
        float(string)
        return True
    except:
        return False
    
    
def list_flattener(nested_list):
    '''Flatten a nested list of any depth to a flat list'''
    if not nested_list:
        return None
    elif type(nested_list[0]) in (list, tuple):
        return list_flattener([n for x in nested_list for n in x])
    else:
        return nested_list
        
        
         
#%%
