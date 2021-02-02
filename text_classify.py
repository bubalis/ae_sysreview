#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:10:16 2021

@author: bdube
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import tensorflow_hub as hub
import numpy as np
from sklearn.decomposition import PCA
import os
import pandas as pd
import seaborn as sns
import utils

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, 
                                 {sentences: x})



embed_fn = embed_useT('text_classify_lite')

def run_partial(data, n_jobs = 10):
    out = []
    part = np.linspace(0, len(data), n_jobs+1)
    for i in range(n_jobs):
        m = embed_fn(data[int(part[i]):int(part[i+1])])
        out.append(m)
    return np.vstack(out)

df = pd.read_csv('/mnt/c/Users/benja/sys_review_dis/data/all_data.csv')

df =df.drop_duplicates(subset = ['AB']).dropna(subset =  ['AB'])
df = utils.filter_relevant(df)

data = df['AB'].tolist()

matrix = embed_fn(data)
#matrix = run_partial(data, 5)

pca = PCA(n_components=3)

pca.fit(matrix)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

reduced = pca.fit_transform(matrix)
reduced = np.ascontiguousarray(reduced)

df['pca_1'] = reduced[:, 0]
df['pca_2'] = reduced[:, 1]
df['pca_3'] = reduced[:, 2]
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sub= df[df['agroecology']==False]
ax.scatter(sub['pca_1'], sub['pca_2'], sub['pca_2'], alpha =.2, s=1)
sub = df[df['agroecology']]
ax.scatter(sub['pca_1'], sub['pca_2'], sub['pca_2'], alpha =1, s=1.5, color='r')
#%%
df =utils.add_topic_cols(df)
sns.scatterplot('pca_1', 'pca_2', 
                hue= 'climate_smart', 
   data=df)
#%%
all_means =[]

columns = ['ecological_intensification',
       'sustainable_intensification', 'climate_smart',
       'sustainable_agriculture', 'agroecology', 'ecoagriculture',
       'alternative_agriculture', 'permaculture', 'ipm', 'biodynamics',
       'regenerative agriculture', 'conservation agriculture',
       'organic agriculture']

for col in columns:
    all_means.append(  df.groupby(col)['pca_1', 'pca_2', 'pca_3'].mean().loc[True])

mdf = pd.DataFrame(columns = ['pca_1', 'pca_2', 'pca_3'])
for item in all_means:
    mdf = mdf.append(item)


mdf.index= columns
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mdf['pca_1'], mdf['pca_2'], mdf['pca_3'])


