#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:24:01 2021

@author: bdube
"""

import pandas as pd
import os
from itertools import permutations, count, takewhile
import utils
from operator import itemgetter
from functools import partial
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from author_work import try_merge
import author_network
from networkx.algorithms import community
from co_occurence_matrix import coocc_matrix, make_cocc_plot

topics = utils.load_topics()
edges = pd.read_csv(os.path.join('data', 'intermed_data', 'ref_matrix.csv'))

art_df = utils.add_topic_cols(pd.read_csv(os.path.join('data', 'intermed_data', 'all_relevant_articles.csv')))

edges = edges.merge(art_df, left_on = 'to', right_on = 'article_id')
edges = edges.merge(art_df, left_on = 'from', right_on = 'article_id', suffixes = ['_source', 'target'])


edges_agg = edges.groupby('author_id_source')[[f'{topic}_source' for topic in topics]+[f'{topic}_target' for topic in topics]].sum()
edges_agg = edges_agg.astype(bool)
coocc = np.array(edges_agg[[f'{topic}_source' for topic in topics]]) @ np.array(edges_agg[[f'{topic}_targe' for topic in topics]]).T
for i in range(coocc.shape[0]):
    coocc[i] = coocc[i]/edges_agg[f'{topics[i]}_source'].sum()

coocc = pd.DataFrame(coocc, index = topics, columns = topics)

make_cocc_plot(coocc, os.path.join('figures', 'corr_citations_article.png'))
coocc.to_csv(os.path.join('data', 'matrixes', 'corr_citations_article.csv')) 
