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

from functools import partial
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from author_work import try_merge
from networkx.algorithms import community
import matrix_utils






def df_2col_dict(df, key_col, val_col):
    return dict(zip(df[key_col], df[val_col]))








'''
def count_overlapping_pubs(aut_row, G, df, pub_cols):
    
    connected_authors = nx.node_connected_component(G, aut_row['author_id'])
    sub = df[df['author_id'].isin(connected_authors)]
    data = sub[pub_cols].max()
    return data - (aut_row[pub_cols]*10000)


for topic in topics:
    print(topic)
    out_data = []
    subset = aut_pubs.loc[aut_pubs[f'num_pubs_{topic}']>0]
    new_cols = [c.replace('num_pubs', 'num_colabs') for c in pub_cols]
    subset[new_cols] = utils.parallelize_on_rows(subset, partial(count_overlapping_pubs, 
                                            G = G, df = aut_pubs, 
                                            pub_cols = pub_cols,)).replace(-9999, np.nan  )
    out_data.append(subset[new_cols].mean())
'''


def edge_df(edges, aut_pubs, id_col= 'author_id'):
    '''Convert a pandas edgelist into a dataframe with author information for 
    source and targe'''
    edges['edge_index'] = edges.index
    edges = edges.merge(aut_pubs, left_on = 'source', right_on = id_col)
    edges = try_merge(edges, aut_pubs, left_on = 'target', right_on = id_col, suffixes = ['_source', '_target'])
    return edges






#%%
def plot_network(g, attribute):
    pos = nx.spring_layout(g)
    node_color = list(nx.get_node_attributes(g, attribute).values())
    nx.draw_networkx_edges(g, pos)
    nc = nx.draw_networkx_nodes(g, pos,
                        node_color = node_color ,  cmap = plt.cm.winter_r, alpha =1, node_size =100)
    plt.colorbar(nc)
    plt.show()


def set_degrees(G):
    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')
    return G
#%%

def extract_subgraphs(G, save_name):
    #if nx.is_directed(G):
    #    G = nx.to_undirected(G)
    sub_graphs = []
    sub_graph_nodes = list(nx.connected_components(G))
    
    print(len(list(sub_graph_nodes)))
    
    for nodes in sub_graph_nodes:
        print(len(nodes))
        sub_graphs.append(G.subgraph(nodes))
    sub_graphs.sort(key = lambda x: len(x.nodes()))    
    main_graph = sub_graphs.pop()
    sub_graph_nodes.sort(key = len)
    communities = community.greedy_modularity_communities(main_graph)
    with open(os.path.join('data', save_name), 'w+') as f:
        for line in [list(c) for c in communities]:
            print('\t'.join(line)+'\n', file = f)
        print('sub graphs:\n', file =f )
        for sg in sub_graph_nodes:
            print('\t'.join(sg)+'\n', file = f)
    return communities
    
if __name__ == '__main__':
    
    adf = pd.read_csv(os.path.join('data', 'intermed_data', 'expanded_authors.csv'))
    
    
    adf.rename(columns ={'ID_num_y': 'article_id'}, inplace=True)
    a_ids = utils.filter_relevant(adf)['author_id']
    adf = adf[adf['author_id'].isin(a_ids)]
    
    
    name_combos = adf.groupby('article_id')['author_id'].agg(list)
    del adf
    
    
    
    aut_edges = [x for y in 
                 name_combos.apply(partial(permutations, r=2)) for x in y]
    
    G =nx.MultiGraph()
    G.add_edges_from(aut_edges)
    
    G = set_degrees(G)
    
    nx.write_edgelist(G, os.path.join('data', 'intermed_data', 'author_edges.edgelist'))
    
    
    aut_pubs = pd.read_csv(os.path.join('data', 'intermed_data', 'author_pub_data.csv'))
    pub_cols = [c for c in aut_pubs.columns if 'num_pubs' in c]
    topics = utils.load_topics()
    for col in pub_cols:
        di = df_2col_dict(aut_pubs, 'author_id', col)
        null_dict = {n: 0 for n in G.nodes() if n not in di}
        di.update(null_dict)
        nx.set_node_attributes(G, di, col)
        
    
    
    del name_combos
    communities = extract_subgraphs(G, 'author_collab_coms_greedy_mod.txt')
    #%%
    
    '''
    com_list=[]
    comp = community.girvan_newman(main_graph)
    limited = takewhile(lambda c: len(c) <= 100, comp)
    i = 0
    for comm_set in limited:
        i+=1
        com_list.append(comm_set)
        print(f'completed set {i} of girvan_newman')
    
    with open(os.path.join('data', 'author_collab_coms_girvan.txt'), 'w+') as f:
        for com in [list(c) for c in comm_set]:
            print(','.join(['\t'.join([c for c in com]) for com in comm_set])+'\n', file = f)
    '''
    
    
    #for SG in sub_graphs:
    #    del SG
    #del sub_graphs
    #del main_graph
    
    
    
    
    edges = nx.to_pandas_edgelist(G)
    del G
    
    
    
    

    aut_pubs[pub_cols] = aut_pubs[pub_cols].astype(bool)
    edges = edge_df(edges, aut_pubs)
    coocc = pd.DataFrame(matrix_utils.make_coocc_autcollab(edges, topics), index = topics).T
    coocc = coocc*matrix_utils.nan_identity(coocc.shape[0])
    coocc.replace({1:np.nan}, inplace = True)
    
    vmax = coocc.max().max()
    ax = sns.heatmap(data=coocc, vmax=vmax, cmap=mpl.cm.cividis_r, linecolor='white',linewidths=1 )
    #ax.xlabels = [l.replace('num_pubs_', '') for l in coocc.columns] 
    plt.savefig(os.path.join('figures', 'author_collab_hm.png'), bbox_inches='tight')
    coocc.to_csv(os.path.join('data', 'matrixes', 'corr_collaboration.csv') )
    
    
    
