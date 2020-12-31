#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:17:21 2020

@author: bdube
"""
import json
import os

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
    while True:
        nested_list=[n for x in nested_list for n in x]
        if type(nested_list[0])!=list:
            return nested_list
#%%
