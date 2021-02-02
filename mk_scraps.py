#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:53:48 2021

@author: bdube
"""
import pandas as pd
import os
from metaknowledge.WOS import  recordWOS
import metaknowledge as mk
from metaknowledge import mkRecord
import utils
import tensorflow as tf
import tensorflow_hub as hub

reader = pd.read_csv('/mnt/c/Users/benja/sys_review_dis/data/all_data_2.csv', chunksize = 50000)

for r in reader:
    df = r
    break


#%%
cols = utils.load_col_names()
cols_to_keep = list(cols.keys())

df = df[[c for c in df.columns if c in cols_to_keep]]


sub = utils.filter_relevant(df)
#%%

K = mk.RecordCollection()
for i, row in df.iterrows():
    R = mkRecord.Record(row.dropna().to_dict(), idValue=row['UT'], bad=False, error=None)
    #print(R)
    K.add(R)
