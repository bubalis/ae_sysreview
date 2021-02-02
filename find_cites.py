# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:17:21 2019

@author: benja
"""


import os
import pandas as pd
import dask.dataframe
import re
from author_work import aut_cleaner
import utils
import numpy as np


#%%
topics= utils.load_topics()

def parse_ref(string):
    keys=['aut', 'year', 'journ', 'DOI', 'vol', 'page']
    pieces = string.split(',')
    dic = {key: None for key in keys}
    if re.search('\d\d\d\d',pieces[0]) or len(pieces) <=2:
        return pd.Series(dic)
    
    dic['aut'] = aut_cleaner(pieces[0]).replace('*', '')
    try:
        dic ['year'] = int(pieces[1].strip())
    except:
        return pd.Series(dic)
    
        
    dic ['journ'] = pieces[2].strip()
    if 'DOI' in pieces[-1]:
        dic ['DOI'] = pieces[-1].replace('DOI ', '').lower().strip(']').strip()
    vol_pieces = [ p for p in pieces if re.search('V\d', p)]
    if vol_pieces:
        dic ['vol'] = vol_pieces[0].replace('V', '').strip()
    page_pieces = [p for p in pieces if re.search('P\d', p)]
    if page_pieces:
        dic ['page'] = page_pieces[0].replace('P', '').strip()
    return pd.Series(dic) 

def get_refs(data):
    '''Get all references from a row reference data column'''
    
    #correction for DOIs with a ';' in them:
    if '[' in data:
        
        problems = re.findall(r'\[.+?\]', data)
        for prob in problems:
            data.replace(prob, prob.replace(';', '*****'))
        return [parse_ref2(ref.replace('*****', ';')) for ref in data.split('; ')]
   
    try:
        return [parse_ref2(ref) for ref in data.split('; ')]
    except:
        print(data)
        raise
        

def ref_df_maker(row):
    '''Complete the dict for parsing ref data '''
    
    refs= get_refs(row['CR'])
    for ref in refs:
        ref.update({'citing_art': row.index})
    return refs

def make_ref_list(string):
    if '[' in string:
        problems = re.findall(r'\[.+?\]', string)
        for prob in problems:
            string.replace(prob, prob.replace(';', '*****'))
    return [x.replace('*****', ';') for x in string.split('; ')]

def make_rdf2(df):
    data= utils.filter_relevant(df).dropna(subset=['CR'])
    data = data[['CR']]
    data['CR'] = data["CR"].apply(make_ref_list)
    data['citing_art'] = data.index  
    data = data.explode('CR')
    data[['aut', 'year', 'journ', 'DOI', 'vol', 'page']] = data['CR'].apply(parse_ref)
                                                           
    data = data.reset_index().drop(columns=['CR', 'index'])
    data['index'] = data.index
    return data

def make_rdf(df):
    '''Make a dataframe of all cited references in dataset.'''
    
    data= utils.filter_relevant(df).dropna(subset=['CR'])
    data = data[['CR']]
    
    return pd.DataFrame(
        utils.list_flattener(
        utils.parallelize_on_rows(
           data, ref_df_maker, num_of_processes=3).tolist()
        )
        )

f_letter =re.compile(r'^.+?\,\s\w')
def if_match(regex, x):
    '''Return the match for a regex if it exists. Else return none.'''
    
    if regex.search(str(x)):
        return regex.search(x).group()
    return None

f_letter_no_com =re.compile(r'^.+?\s\w')


def aut_ln_fi(stri):
    if not stri:
        return None
    res = if_match(f_letter, stri)
    if res:
        return res.lower()
    return None

def ref_aut_ln_fi(stri):
    res = if_match(f_letter_no_com, stri)
    if res:
        return res.replace(' ', ', ').lower()
    return None

def vol_formatter(data):
    '''Correction for the Pub Volume Column'''
    
    return str(data).replace('.0', '')


def lower_if(string):
    '''If string is actually a string, return it lowercase.'''
    if type(string) == str:
        return string.lower()
    
    
def format_article_idnum(df):
    '''Give a unique article number to every line in the dataframe.
    If it has a DOI, this is used. If no DOI is available, use other 
    information in the dataframe.'''
    
    df['ID_num'] = None
    df.loc[df['ID_num'].isnull(), 'ID_num'] = df['TI'].apply(lambda x: x[3:] ) +'::' + df.index.astype(str)
    df.loc[df['DI'].isnull() ==False, 'ID_num'] = df['DI'].dropna()
    df.loc[
        ((df['DI'].isnull()) & 
         (df['SN'].isnull() == False) & 
         (df['PT'].isin('B', 'S'))), 'ID_num'
        ] = 'book chap ' + df['SN'] + ' // ' +df.index.astype(str)
    
  
    df.loc[(df['ID_num'].isnull()) & (df['UT'].isnull()==False), 'ID_num'] = df[(df['ID_num'].isnull()) & 
                                                 (df['UT'].isnull() ==False)]['UT']
    
    df.loc[df['ID_num'].isnull(), 'ID_num'] = df['TI'].apply(lambda x: x[3:] ) +'::' + df.index.astype(str)
    
    #%%
def collect_cites(df, rdf):
    doi_corrections = []
    
    rdf['DOI'] = rdf["DOI"].apply(lower_if)
    
    df['DI'] = df['DI'].apply(lower_if)
    
    match_sets =[]
    #perfect DOI matches:
    matches = df.dropna(subset=['DI']).merge(
                rdf, left_on = 'DI', 
                right_on = "DOI", how = 'inner')
    
    matches = matches[['DOI', 'DI', 'citing_art', 'index_x', 'index_y',
                                                  #'J9', 
                                                  #'PY', 'journ', 
                                                  #'year', 'VL', 'vol',
                                                  #'page', 'BP'
                                                  ]]
    
    assert matches['index_x'].isin(df['index']).all()                                            
    match_sets.append(matches)
    
    
    #df.dropna(subset= ['AU'], inplace = True)
    df['ln_fi'] = df['AF'].apply(aut_ln_fi)
    rdf['ln_fi'] = rdf['aut'].apply(ref_aut_ln_fi)
    df['VL'] = df['VL'].apply(vol_formatter)
    
    #get matches where ln_fi of first author, journal name and year are all same
    #try with 4 different columns where the journal name might be
    for journ_col in ['J9', 'SO', 'JI', 'SO', 'PU']:
        
        m = df.dropna(subset=['ln_fi']).merge(rdf, 
                    left_on =   ['ln_fi', journ_col, 'PY'], 
                     right_on = ['ln_fi', 'journ', 'year'], 
                     how ='inner')
        
        m.dropna(subset = ['ln_fi', 'journ', 'year'], how ='any',
                 inplace= True)
        #subset these matches for where the citation volume or page number =
        #that of the possible article match, or both of these are missing. 
        m = m[((m['VL'] == m['vol']) | (m['vol'].isnull()))  
              & 
              ((m['BP'] == m['page']) | (m['page'].isnull()))
              ]
        
        print(f'{m.shape[0]} matches found')
        assert 'index' not in m.columns
        try:
            if ((m['DOI'].isnull()) & (m['DI'].isnull()==False)).sum() > 0:
                m.loc[(m['DOI'].isnull()) & (m['DI'].isnull()==False), 
                      'DOI'] = m['DI'].dropna()
                
                
                doi_corrections.append(m.loc[(m["DI"].isnull()) & 
                                             (m['DOI'].isnull()==False),['DOI', 'index_x']])
            
            assert m['index_x'].isna().sum()== 0
            assert m['index_x'].isin(df['index']).all()
            
            match_sets.append(m[['DOI', 'DI', 'citing_art', 
                                 'index_x', 'index_y']])
        except:
            globals().update(locals())
            assert 'm' in globals()
            raise
    try:
        assert all ([p['index_x'].isin(df['index']).all() for p in match_sets])
        refs = pd.concat(match_sets)
        assert refs['index_x'].isin(df['index']).all()
    except:
        globals().update(locals())
        raise
    doi_corrections = pd.concat(doi_corrections).drop_duplicates()
    return refs, doi_corrections
#%%
def correct_dois(df, doi_corrections):
    '''Fix DOIs that had been missing from the original article.'''
    try:
        
        df = df.merge(doi_corrections, left_on = 'index', 
                  right_on= 'index_x', 
                  how='left')
    
        df.loc[(df['DI'].isnull()) 
               & (df['DOI'].isnull()==False), "DI"] =df['DOI']
    
        return df.drop(columns=['DOI'])
    
    except:
        globals().update(locals())
        raise
        
        
def give_IDnums(df):
    df['ID_num'] = np.nan
    df.loc[df['ID_num'].isnull(), 'ID_num'] = df['TI'].apply(lambda x: x[3:] ) +'::' + df.index.astype(str)
    df.loc[df['DI'].isnull() ==False, 'ID_num'] = df['DI'].dropna()
    df.loc[
        ((df['DI'].isnull()) & 
         (df['SN'].isnull() == False) & 
         (df['PT'].isin(['B', 'S']))), 'ID_num'
        ] = 'book chap ' + df['SN'] + ' // ' +df['index'].astype(str)
    
  
    df.loc[(df['ID_num'].isnull()) & (df['UT'].isnull()==False), 'ID_num'] = df[(df['ID_num'].isnull()) & 
                                                (df['UT'].isnull() ==False)]['UT']
    
    df.loc[df['ID_num'].isnull(), 'ID_num'] = df['TI'].apply(lambda x: x[3:] ) +'::' + df.index.astype(str)
    return df



#%%

if __name__=='__main__':
    path=os.path.join('data', 'all_data.csv')
    df = pd.read_csv(path)
    test = False
    if test:
        df = df.head(50000)
    df.drop(columns = ['Unnamed: 0', 'level_0', 'index'], inplace=True)
    df['index'] = df.index
    assert df['index'].isna().sum() == 0
    print(df.shape)
    print('Assembling DataFrame of References')
    rdf = make_rdf2(df)
    
    print('Finding Citations')
    refs, doi_corrections = collect_cites(df, rdf)
    
    print('Correcting DOIs based on those found in references')
    
    
    df = correct_dois(df, doi_corrections)
    
    print('Assigning ID numbers to articles.')
    df = give_IDnums(df)
    
    
    refs =  refs.merge(df[['ID_num', 'index']], left_on = 'index_x', 
                       right_on='index',
                       how ='left')
    
    refs = refs.merge(df[['ID_num']], left_on = 'citing_art', 
                      right_index = True, how = 'left')
    #refs.drop(columns = ['citing_art'], inplace= True)
    refs = refs.rename(columns ={'ID_num_x': 'ID_num', 
                                 'ID_num_y': 'citing_article' } 
                      )
    
    
    refs.to_csv(os.path.join('data', 'ref_matrix.csv'))
    
    missing = rdf.dropna(subset= ['DOI'])[rdf['DOI'].dropna().isin(
                        refs['ID_num'])==False]
    missing.to_csv(os.path.join('data', 'out_sample_refs.csv'))
    missing['DOI'].value_counts().to_csv(os.path.join('data', 'out_sample_refs_counts.csv'))
    df.to_csv(os.path.join('data', 'all_data_2.csv'))
    
    pd.concat([utils.filter_relevant(df), 
               df[df['ID_num'].isin(refs['ID_num'])]]
              ).drop_duplicates().to_csv('all_relevant_articles.csv')
    
    
   
    