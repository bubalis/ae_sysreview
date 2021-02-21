#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:24:01 2021

@author: bdube
"""

import pandas as pd
import os
import utils
import matrix_utils
from matrix_utils import only_topic_cols
from utils import topics



#%%
if __name__ == '__main__':
    edges = pd.read_csv(os.path.join('data', 'intermed_data', 'ref_matrix.csv'))
    edges = edges.rename(columns = {'from': 'source', 'to': 'target'})
    art_df = utils.add_topic_cols(pd.read_csv(os.path.join('data', 'intermed_data', 'all_relevant_articles.csv')))
    
    edges = edges.merge(art_df, left_on = 'source', right_on = 'ID_num')
    edges = edges.merge(art_df, left_on = 'target', right_on = 'ID_num', suffixes = ['_source', '_target'])
    edges = edges.groupby('source')[only_topic_cols(edges, topics)].sum().astype(bool)
    
    del art_df
    
    coocc = matrix_utils.network_coocc(edges, list(topics.keys()))
    
    matrix_utils.cocc_plot(coocc, os.path.join('figures', 'corr_citations_article.png'))
    coocc.to_csv(os.path.join('data', 'matrixes', 'corr_citations_article.csv')) 
