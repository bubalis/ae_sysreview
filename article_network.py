#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:35:32 2021

@author: bdube
"""

import os
import networkx as nx
import pandas as pd

from networkx.algorithms import community




G = nx.read_edgelist(open(os.path.join('data', 'intermed_data', 'ref_matrix.csv'), 'rb'))

df = pd.read_csv(os.path.join('data', 'intermed_data', 'ref_matrix.csv'))
tos= df['to'].tolist()
froms = df['from'].tolist()


G= nx.DiGraph()
G.add_edges_from([(to, _from) for to, _from in zip(tos, froms) ])

degree_dict = dict(G.in_degree(G.nodes()))
nx.set_node_attributes(G, degree_dict, 'degree')
sorted_degree = sorted(degree_dict.items(), key=itemgetter(1), reverse=True)

print("Top 20 nodes by in degree:")
for d in sorted_degree[:20]:
    print(d)