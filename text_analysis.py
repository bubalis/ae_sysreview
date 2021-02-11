#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:38:46 2021

Script for comparing text of abstracts within the dataset.

@author: bdube
"""

import os 
import pandas as pd
import utils
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import nltk
import string
import seaborn as sns
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import wordnet, stopwords
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


topic_abbrvs = ['EI', 'CA', 'SI', 'CS', 'SA', 'AE', "EA", 'AA', 
                "PC", 
                #"IPM", 
                'BD', 'RA', 'OA']

def get_wordnet_pos(treebank_tag):
    '''Turn a treebank tag into a wordnet pos tag.
    Used for the WNL lemmatizer'''
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

wnl = WordNetLemmatizer()
def lemmatize(word, pos):
    if pos:
        return wnl.lemmatize(word,pos)
    else:
        return word
    


parentheses= ['(', ')', '"', "'", ',', '.', r'\\', ]

#%%
ps = PorterStemmer()

def prep_document(doc, Stemmer=PorterStemmer):
    '''Pre-process a the document for performing word counts.'''
    
    
    stemmer = Stemmer()
    #remove text after copyright:
    doc = doc.split('(C)')[0]
    
    #remove numbers:
    doc = ' '.join([word for word in doc.split(' ') if all(
        [c not in '1234567890' for c in word])])
    
    #remove symbols
    doc = ''.join([c for c in doc if c in 
                   string.ascii_letters+string.whitespace+'-'])
    
    #turn dashes into whitespace  
    doc = doc.replace('-', ' ')
    
    #stem all words
    return ' '.join([stemmer.stem(w) for w in doc.split(' ')])
    

    
def prep_all_docs(corpus, Stemmer= LancasterStemmer):
    return [prep_document(doc, Stemmer) for doc in corpus]



docPrep = FunctionTransformer(prep_all_docs)









def xform_data(corpus, stop_words):
    '''Preprocess data, and fit it to corpus.
    Return the transformed data and the word list'''
    
    pipe = Pipeline([('prep', docPrep),
        ('count', CountVectorizer(stop_words = stop_words)),
                   ('tfid', TfidfTransformer())
        ]).fit(corpus)
    
    data = pipe.transform(corpus)
    words = pipe['count'].get_feature_names()
    
    return data, words


def test():
    stemmer = LancasterStemmer()

    topics = ['Feminist Geography', 
                'High Energy Physics', 
                'Virology'
                ]
    topic_words = [x.split(' ') for x in topics]
    stop_words = stopwords.words('english')+ [x for y in topic_words for x in y]

    stop_words+=[stemmer.stem(word) for word in stop_words]
    
    t_dir= os.path.join('data', 'toy')
    df1 = load_test_data(os.path.join(t_dir, 'fem_geo.txt'))
    df2 = load_test_data(os.path.join(t_dir, 'h_2_phys.txt'))
    df3 = load_test_data(os.path.join(t_dir, 'viral.txt'))
    corpus = df1['ID'].tolist()+ df2['ID'].tolist() + df3['ID'].tolist()
    #corpus = [prep_document(d) for d in corpus]
    splits = [df1.shape[0], 
              df1.shape[0]+df2.shape[0], 
              len(corpus)
              ]
   
   
    data, t = xform_data(corpus, stop_words)
    agg_arrays = make_category_mean_arr(data.toarray(), splits)
    pca =PCA(n_components =2)
    pca.fit(agg_arrays)
    a = data.toarray()
    arrays = make_category_mean_arr(data.toarray(), splits)
    pca =PCA(n_components =2)
    
    pca.fit(arrays)
    
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    
    arrays2 = make_category_mean_arr(data.toarray(), splits, 
                                    test=True)
    
    pca2 =PCA(n_components =2)
    pca2.fit(arrays2)
    print('Results for randomly shuffled data:')
    print(pca2.explained_variance_ratio_)
    print(pca2.singular_values_)
    
    pca3 =PCA(2)
    
    xform=pca.transform(data.toarray())
    
    topics = ['Feminist Geography', 
                    'High Energy Physics', 
                    'Virology'
                    ]
    
    
    
    s = [0]+splits
    for i in range(len(splits)):
        data_to_plot = xform[s[i]:s[i+1]]
        plt.scatter(data_to_plot[:, 0], data_to_plot[:, 1], alpha=.3)
        plt.legend(topics
                   )
    plt.show()
    
    xform2 = pca2.transform(data.toarray())
    for i in range(len(splits)):
        data_to_plot = xform2[s[i]:s[i+1]]
        plt.scatter(data_to_plot[:, 0], data_to_plot[:, 1], alpha=.3)
        plt.legend(topics
                   )
    plt.show()
    
    pca3.fit(data.toarray())
    
    xform3 = pca3.transform(data.toarray())
    
    for i in range(len(splits)):
        data_to_plot = xform3[s[i]:s[i+1]]
        plt.scatter(data_to_plot[:, 0], data_to_plot[:, 1], alpha=.3)
        plt.legend(topics
                   )
    plt.show()
   
    
def load_test_data(path):
    '''Load in data for test.'''
    df = pd.read_csv(path, sep= '\t')
    df['ID'] = df['ID'].replace('', np.nan)
    return df.dropna(subset=['ID'])







def make_category_mean_arr(a, splits, test=False):
    '''Pass an 2d-array and splits idicies (row numbers). 
    Return an array of category means for each column.'''
    #Random shuffle is used to test: category means should be meaningless
    if test:
        np.random.shuffle(a)
    
    arrays = []
    s = [0]+splits
    for i in range(len(splits)):
        arrays.append(a[s[i]:s[i+1]].mean(axis =0))
    return np.stack(arrays)
    



#%%
def run(indf, data_col, topics, stemmer=LancasterStemmer(), n_pca_components = 2,
        fit_to_all = False, 
        extra_corpus = False):
    
    indf=indf[(indf[topics].sum(axis=1)==1) | (indf[topics].sum(axis=1)==0)] # Only abstracts assigned to exactly 1 keyword
    
    corpus, splits = load_text_data(indf, data_col, 
                                    topics, whole_dataset = extra_corpus)
    del indf
    
    
    stop_words = set_stopwords(topics, stemmer)
    
    data, words = xform_data(corpus, stop_words)
    #agg_arrays = make_category_mean_arr(data.toarray(), splits)
    
    if extra_corpus:
        pca =TruncatedSVD(n_components = n_pca_components)
        pca.fit(data)
        name = 'fit_to_all'
        data = data[:splits[-2]]
        _ = splits.pop()
        agg_arrays = make_category_mean_arr(data, splits)
        data = data.toarray()
        print(f'Shape of data: {data.shape}')
    else:
        pca =PCA(n_components = n_pca_components)
        data = data.toarray()   
        agg_arrays = make_category_mean_arr(data, splits)
        if fit_to_all:
            pca.fit(data)
            name = 'fit_to_all'
        else:
        
            pca.fit(agg_arrays)
            name = 'standard'
    
    
    outdf = analyze(data, pca, topics, splits, corpus, agg_arrays, name) 
    
        
    
    return outdf, pca, agg_arrays, words


def analyze(data, pca, topics, splits, corpus, agg_arrays, name):
    
    
    out_df = make_df(data, agg_arrays, pca, topics, splits)
    out_df['text'] = corpus[:splits[-1]]

    #g = sns.scatterplot(x='pca_1', y = 'pca_2', data=out_df, hue = 'topic', alpha =.3, s=2)
    #plt.legend([],[], frameon=False)
    #plt.savefig(os.path.join('figures', f'pca_{name}.png'))
    
    #legend = plt.legend(topics)
    #plt.show()
    #fig  = legend.figure
    #fig.canvas.draw()
    #plt.savefig(os.path.join('figures', 'legend'))
    #plt.show()
    return out_df


def make_df(a, agg_arrays, pca, topics, splits):
    '''Collate data from the pca process into a dataframe.
    columns = 
    pca_1: score on first component of pca
    pca_2: score on 2nd component of pca
    sim_score_x: for each topic group, what is the sim
    topic
    '''

    
    results = []

        
    try:
        df = pd.DataFrame(cosine_similarity(a, agg_arrays), 
                          columns = [f'sim_score_{t}' for t in topics])
        
        #for i in range(len(topics)):
            #results.append(np.apply_along_axis(cosine_sim, 1, a, np.array(agg_arrays[i])[0]))
        
        #df =pd.DataFrame(np.array(results).T, columns = [f'sim_score_{t}' for t in topics])
        
        xform=pca.transform(a)
        del a 
        for i in range(pca.n_components):
            df[f'pca_{i+1}'] = xform[:, i]
            #df[f'pca_{i+1}_exp']=np.exp( df[f'pca_{i+1}'])
            
        s = [0]+splits
        for i, topic in zip(range(len(splits)), topics):
            df.loc[s[i]:s[i+1], 'topic'] = topic
    except:
        globals().update(locals())
        raise
    return df 
    
    
    
       
def load_text_data(indf, data_col, topics, whole_dataset=False):
    '''Load textual data from the indf. 
    _data_col_ : column with the text data.
    _topics_: names of boolean columns that contain whether the data fit into 
    a given sub-corpus.
    
    returns: 
        corpus: a list of documents.
        splits: the indicies where the sub-corpuses change. 
        '''
    
    corpus = []
    splits = []
    
    #add data to the corpus by topic
    for t in topics:
       data=indf[indf[t].astype(bool)][data_col]
       data = data.replace('', np.nan).dropna()
       corpus+= data.tolist()
       splits.append(len(corpus))
    
    #if 
    if whole_dataset:
        data = indf[indf['any_topic']==0][data_col]
        data = data.replace('', np.nan).dropna()
        corpus+= data.tolist()
        splits.append(len(corpus))
        
    
       
    return corpus, splits


def set_stopwords(topics, stemmer):
    '''Set stopwords based on topics and stemmer.
    This function adds the topic keywords to the stop words and 
    applies the stemmer to the stopwords as well'''
    topic_words = [re.split('[\s_]', x) for x in topics]
    stop_words = stopwords.words('english')+ [x for y in topic_words for x in y]+['integrated', 'pest', 'management']
    return stop_words + [stemmer.stem(word) for word in stop_words]

def cosine_sim(a1, a2):
    '''Return the cosine similarity of two vectors.'''
    return np.dot(a1, a2)/ (np.linalg.norm(a1) * np.linalg.norm(a2))




def get_most_sig_words(pca, index, word_list, percentile=.999):
    '''Return the most important words based on the PCA.
    
    _pca_ : the fitted pca. 
    _index_: the component number of the pca
    _word_list_: list of words.
    _percentile: what fraction of words to return. Default: .999, or the 
    .001% most significant words.'''
    
    
    a = pca.components_[index]
    a = a - a.mean()
    absolut = np.abs(a)
    indexes = np.where(absolut>np.quantile(absolut, percentile))[0].tolist()
    out = [(word_list[i], a[i]) for i in indexes]
    return sorted(out, key= lambda x: x[1])






def pairwise_compare_on_avg(df, topics):
    '''Turn similarity score columns of dataframe into a matrix.
    average score of all documents in _row_ with the average document in _column_'''
    out_array=[]
    for topic in topics:
        sub_array=[]
        sub= df[df['topic']==topic]
        for t in topics:
            sub_array.append(sub[f'sim_score_{t}'].mean())
        out_array.append(sub_array)
    
    return np.array(out_array)


def avg_cosine_sim(a, splits):
    '''This is probably a bad idea.'''
    out = []
    c_a = cosine_similarity(a)
    s = [0] + splits
    for i in range(len(splits)):
        row = []
        for i2 in range(len(splits)):
            row.append(c_a[s[i]:s[i+1], s[i2]: s[i2+1]].mean())
        out.append(row)

#%%
def load_from_df(path, extra_corpus):
    '''Load in a dataframe from path.'''
    df = pd.read_csv(path)
    print(df.shape)
    df = utils.add_topic_cols(df)
    if not extra_corpus:
        df = df[df['any_topic'].astype(bool)] #filter to only relevant data points
    print(df.shape)
    return df

    

word_data_dir = os.path.join('data', 'important_words')

def main(data_path, fit_to_all, extra_corpus, 
         n_pca_components =2):
    '''Run all text analysis.
    args: data_path: path to csv file of corpus.
    
    fit_to_all: (bool) does the decomposition algorithm fit to all abstracts, 
    or just the aggregates?
    
    extra_corpus: (bool) is a larger corpus, not to be analyzed, loaded, 
    to fit decomposition algorithm and to identify word counts?
    
    n_pca_components: (int) how many components to decompose into?'''
    
    if extra_corpus:
        name = 'based_on_all_data'
    else:
        name = 'just_sample'
    
    df = load_from_df(data_path, extra_corpus)
    topics = list(utils.load_topics().keys())
    #out_df, pca, agg_arrays, words = run(df, 'AB', topics)
    #compare_matrix = pairwise_compare_on_avg(out_df, topics)
    #sns.heatmap(compare_matrix, cmap= 'winter_r')
    #plt.savefig(os.path.join('figures', 'textual_heatmap.png'))
    #plt.close()
    #del out_df
    #del pca 
    #del words
    #del compare_matrix
    out_df, pca, agg_arrays, words = run(df, 'AB', topics, 
                                         fit_to_all=fit_to_all, 
                                         n_pca_components = n_pca_components, 
                                         extra_corpus = extra_corpus)
    
    compare_matrix = pairwise_compare_on_avg(out_df, topics)
    ax = sns.heatmap(compare_matrix, cmap= 'winter_r')
    ax.set_xticklabels(topic_abbrvs)
    plt.xticks(rotation=90)
    ax.set_yticklabels(topic_abbrvs)
    plt.yticks(rotation=90)
    plt.savefig(os.path.join('figures', f'textual_heatmap_{name}.png'))
    plt.show()
    compare_matrix2 = cosine_similarity(agg_arrays)
    ax = sns.heatmap(compare_matrix2, cmap= 'winter_r')
    ax.set_xticklabels(topic_abbrvs)
    ax.set_yticklabels(topic_abbrvs)
    plt.savefig(os.path.join('figures', f'textual_heatmap_aggregate_{name}.png'))
    plt.show()
   
    
    
    plot_vars = [c for c in out_df.columns if 'pca_' in c]
    plot_df = out_df[out_df['topic'].isin(out_df['topic'].value_counts().head(3).index)]
    sns.pairplot(plot_df, vars = plot_vars, hue = 'topic',
                 plot_kws={'s': 5, 'alpha' :.1})
    plt.savefig(os.path.join('figures', f'pair1_{name}.png'))

    plot_df = out_df[out_df['topic'].isin(['agroecology', 'sustainable_intensification', 
                                           'conservation agriculture', 'climate_smart'])]
    sns.pairplot(plot_df, vars = plot_vars, hue = 'topic',
                 plot_kws={'s': 10, 'alpha' :.4})
    plt.savefig(os.path.join('figures', f'pair2_{name}.png'))
    
    
 
    plot_df = out_df[out_df['topic'].isin(['ecological_intensification', 'ecoagriculture',
     'alternative_agriculture',
     'permaculture',
     'biodynamics',
     'regenerative agriculture'])]
    
    sns.pairplot(plot_df, vars = plot_vars, hue = 'topic',
                 #plot_kws={'s': 10, 'alpha' :.4}
                 )
    plt.savefig(os.path.join('figures', f'pair3_{name}.png'))
    
    for i in range(n_pca_components):
        with open(os.path.join(word_data_dir, f'{name}_words_{i}.txt'), 'w+') as f:
            word_list = get_most_sig_words(pca, i, words)
            for line in word_list:
                print (f"{line[0]} :   {line[1]}\n", file = f)
    
    np.savetxt(os.path.join(word_data_dir, f'textual_heatmap_{name}.csv'), 
                  compare_matrix, delimiter=",")
    np.savetxt(os.path.join(word_data_dir, f'textual_heatmap_aggregate_{name}.csv'), 
                  compare_matrix2, delimiter=",")

    
    print('Explained Variance of PCA components:')
    print(pca.explained_variance_ratio_)
    
    return pca, out_df, word_list
    
if __name__=='__main__':
    
    if not os.path.exists(word_data_dir):
       os.makedirs(word_data_dir)
       
    path = os.path.join('data', 'intermed_data',  'all_data_2.csv')
    fit_to_all = True
    extra_corpus = False
    _ = main(path, fit_to_all, extra_corpus, n_pca_components = 2)
    del _
    
    #path2 = os.path.join('data', 'intermed_data','all_relevant_articles.csv')
    
    #pca, out_df, word_list = main(path2, fit_to_all=True, 
    #                             extra_corpus=True, n_pca_components = 6)

#%%
