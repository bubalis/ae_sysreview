#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:03:57 2021

@author: bdube
"""
import pandas as pd
import os
import utils
import networkx as nx
import author_network
import matrix_utils
from matrix_utils import only_topic_cols
from utils import topics



if __name__ == '__main__':
    
    
    topics = list(topics.keys())
    
    
    edges = pd.read_csv(os.path.join('data', 'intermed_data', 'ref_matrix.csv'))
    adf = utils.add_topic_cols(pd.read_csv(
        os.path.join('data', 'intermed_data', 'expanded_authors.csv'))
    )
    adf = adf[['author_id', 'ID_num_y']+only_topic_cols(adf, topics)]
    
    
    
    edges = edges.rename(columns = {'from': 'source', 'to': 'target',
                                    'ID_num_y': 'article_id'})
    adf.rename(columns = {'ID_num_y': 'article_id'}, inplace= True)
    edges = author_network.edge_df(edges, adf, id_col = 'article_id')
    edges = edges[['author_id_source', 'author_id_target', 
                   'article_id_source', 
                   'article_id_target']+only_topic_cols(edges, topics)]
    
    
    edges[only_topic_cols(edges, topics) ] = edges[only_topic_cols(edges, topics) ].astype(bool)
    edges.rename(columns = {c: c.replace('author_id_', '') for c in edges.columns}, inplace= True)
    
    coocc = matrix_utils.network_coocc(edges, topics)
    matrix_utils.cocc_plot(coocc, os.path.join('figures', 'corr_citations_author.png'))
    coocc.to_csv(os.path.join('data', 'matrixes', 'corr_citations_author.csv')) 
    
    del adf
    
    G = nx.MultiDiGraph()
    aut_edges = list(zip(edges['source'], edges['target']))
    
    G.add_edges_from(aut_edges)
    communities = author_network.extract_subgraphs(nx.to_undirected(G), save_name = 'author_cite_greedy_mod.txt')
    
    #adf2 = pd.read_csv(os.path.join('data', 'intermed_data', 'all_author_data.csv'))
    
    #edges = edge_df(edges, adf2)
