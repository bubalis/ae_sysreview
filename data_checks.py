#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:59:04 2021

@author: bdube
"""
for file in bad_dfs:
    df = reload_df(file)
    if not df_checker(df):
        print('Failed Condition 1')
    if not df_checker2(df):
        print('Failed Condition 2')
    if not df_checker3(df):
        print('Failed Condition 3')
        
    if not check_df_dtypes(df):
        print('DTypes Failed')
    else:
        print('Passed All Conditions')

    print(file)
    continue
