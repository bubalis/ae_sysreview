# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:17:21 2019

@author: benja
"""


import os
import pandas as pd
import re
from author_work import aut_cleaner
import utils
import numpy as np


#%%
topics= utils.load_topics()

def parse_ref(string):
    '''Parser for reference sub-strings.
    Returns a pd series with labels: 
        ['aut', 'year', 'journ', 'DOI', 'vol', 'page']'''
    
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


        
def make_ref_list(string):
    '''Convert string into a list of reference strings.
    '''
    
    #correction for dealing with dois with ';' in the them:
    if '[' in string:
        problems = re.findall(r'\[.+?\]', string)
        for prob in problems:
            string.replace(prob, prob.replace(';', '*****'))
            
    return [x.replace('*****', ';') for x in string.split('; ')]


def make_rdf(df):
    '''Make a dataframe of all relevant articles in the dataframe.
    columns : ['aut', 'year', 'journ', 'DOI', 'vol', 'page', 'citing article' and 'index']'''
    data= utils.filter_relevant(df).dropna(subset=['CR'])
    data = data[['CR']]
    data['CR'] = data["CR"].apply(make_ref_list)
    data['citing_art'] = data.index  
    data = data.explode('CR')
    data[['aut', 'year', 'journ', 'DOI', 'vol', 'page']] = data['CR'].apply(parse_ref)
                                                           
    data = data.reset_index().drop(columns=['CR', 'index'])
    data['index'] = data.index
    return data


f_letter =re.compile(r'^.+?\,\s\w')
def if_match(regex, x):
    '''Return the match for a regex if it exists. Else return none.'''
    
    if regex.search(str(x)):
        return regex.search(x).group()
    return None

f_letter_no_com =re.compile(r'^.+?\s\w')


def aut_ln_fi(stri):
    '''Get the lastname, first-initial combo for an author.'''
    if not stri:
        return None
    res = if_match(f_letter, stri)
    if res:
        return res.lower()
    return None

def ref_aut_ln_fi(stri):
    '''Get the lastname, first-initial combo for an author, correcting
    for common formatting snafu. '''
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
    
    return df
    #%%
def collect_cites(df, rdf):
    '''Match citations in the rdf to articles in the main dataframe.
    Returns a dataframe of references found, and DOI corrections - dois to add to
    the main dataframe that are missing in that article's metadata. '''
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
    for journ_col in ['J9', 'SO', 'JI', 'PU']:
        
        m = df.dropna(subset=['ln_fi']).merge(rdf, 
                    left_on =   ['ln_fi', journ_col, 'PY'], 
                     right_on = ['ln_fi', 'journ', 'year'], 
                     how ='inner')
        
        m.dropna(subset = ['ln_fi', 'journ', 'year'], how ='any',
                 inplace= True)
        
        #subset these matches for where the citation volume or page number 
        #either are the same as that of the possible article match
        # or both of these are missing. 
        m = m[((m['VL'] == m['vol']) | (m['vol'].isnull()))  
              & 
              ((m['BP'] == m['page']) | (m['page'].isnull()))
              ]
        
        print(f'{m.shape[0]} matches found')
        assert 'index' not in m.columns
    
        if ((m['DOI'].isnull()) & (m['DI'].isnull()==False)).sum() > 0:
           
            m.loc[(m['DOI'].isnull()) & (m['DI'].isnull()==False), 
                  'DOI'] = m['DI'].dropna()
            
            #add 
            doi_corrections.append(m.loc[(m["DI"].isnull()) & 
                                         (m['DOI'].isnull()==False),
                                         ['DOI', 'index_x']])
        else:
            print("All cites have a valid DOI")
        
        assert m['index_x'].isna().sum()== 0
        assert m['index_x'].isin(df['index']).all()
        
        match_sets.append(m[['DOI', 'DI', 'citing_art', 
                             'index_x', 'index_y']])
        
    try:
        assert all ([p['index_x'].isin(df['index']).all() for p in match_sets])
        
        refs = pd.concat(match_sets).drop_duplicates()
        
        assert refs['index_x'].isin(df['index']).all()
    except:
        globals().update(locals())
        raise
    
    doi_corrections = pd.concat(doi_corrections).drop_duplicates()
    
    
    return refs, doi_corrections
#%%
def correct_dois(df, doi_corrections):
    '''Add DOIs that were 'discovered' in the matching process. '''
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
    '''Assign ID numbers to all articles in the dataframe.'''
    
    df['ID_num'] = np.nan
    
    
    df.loc[df['DI'].isnull() ==False, 'ID_num'] = df['DI'].dropna()
    
    
    df.loc[
        ((df['DI'].isnull()) & 
         (df['SN'].isnull() == False) & 
         (df['PT'].isin(['B', 'S']))), 'ID_num'
        ] = 'book_chap ' + df['SN'] + ' // ' +df['index'].astype(str)
    
  
    df.loc[(df['ID_num'].isnull()) & (df['UT'].isnull()==False), 'ID_num'] = df[(df['ID_num'].isnull()) & 
                                                (df['UT'].isnull() ==False)]['UT']
    
    left =  df.loc[df['ID_num'].isnull()]
    df.loc[df['ID_num'].isnull(), 'ID_num'] = left['TI'].apply(lambda x: x[:4] ) +'::' + left.index.astype(str)
    
    
    

    
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
    rdf = make_rdf(df)
    
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
    
   
    
    refs[['ID_num', 'citing_article']].rename(
        columns =
        {'ID_num': 'to',
         'citing_article' : 'from'}
        ).to_csv(os.path.join('data', 
                              'intermed_data', 
                              'ref_matrix.csv'))
    
    missing = rdf.dropna(subset= ['DOI'])[rdf['DOI'].dropna().isin(
                        refs['ID_num'])==False]
    
    #save list of refs outside of sample
    missing.to_csv(os.path.join(
        'data', 'intermed_data', 'out_sample_refs.csv'))
    
    
    missing['DOI'].value_counts().to_csv(os.path.join('data', 'out_sample_refs_counts.csv'))
    df.to_csv(os.path.join('data', 'intermed_data', 'all_data_2.csv'))
    
    pd.concat([utils.filter_relevant(df), 
               df[df['ID_num'].isin(refs['ID_num'])]]
              ).drop_duplicates().to_csv(os.path.join('data', 
                                                      'intermed_data',
                                                      'all_relevant_articles.csv'))
    
    
   
    