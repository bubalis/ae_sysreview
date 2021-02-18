#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:03:57 2021

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

edges = pd.read_csv(os.path.join('data', 'intermed_data', 'ref_matrix.csv'))

adf = pd.read_csv(os.path.join('data', 'intermed_data', 'expanded_authors.csv'))

edges = edges.rename(columns = {'from': 'source', 'to': 'target'})
edges = author_network.edge_df(edges, adf)
edges = edges[['author_id_source', 'author_id_target']]
edges.rename(columns = {c: c.replace('author_id_', '') for c in edges.columns})

del adf

G = nx.MultiDiGraph()
aut_edges = list(zip(edges['source'], edges['target']))

G.add_edges_from(aut_edges)


#adf2 = pd.read_csv(os.path.join('data', 'intermed_data', 'all_author_data.csv'))

#edges = edge_df(edges, adf2)
