#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:01:16 2021

@author: bdube
"""
import pandas as pd
import os
from itertools import permutations
import utils
from operator import itemgetter
from functools import partial
import networkx as nx

adf = pd.read_csv(os.path.join('data', 'intermed_data', 'expanded_authors.csv'))


adf.rename(columns ={'ID_num_y': 'article_id'}, inplace=True)
a_ids = utils.filter_relevant(adf)['author_id']
adf = adf[adf['author_id'].isin(a_ids)]


name_combos = adf.groupby('article_id')['author_id'].agg(list)
del adf



aut_edges = [x for y in name_combos.apply(partial(permutations, r=2)) for x in y]

G =nx.Graph()
G.add_edges_from(aut_edges)

degree_dict = dict(G.degree(G.nodes()))
nx.set_node_attributes(G, degree_dict, 'degree')
sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)


print("Top 20 nodes by degree:")
for d in sorted_degree[:20]:
    print(d)

nx.is_connected(G)


nx.write_edgelist(G, os.path.join('data', 'intermed_data', 'author_edges.edgelist'))
