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



#%%
topics= utils.load_topics()

def parse_ref2(string):
    pieces = string.split(',')
    dic = {key: None for key in ['aut', 'year', 'journ', 'DOI', 'vol', 'page']}
    if re.search('\d\d\d\d',pieces[0]) or len(pieces) <=2:
        return dic
    
    dic['aut'] = aut_cleaner(pieces[0]).replace('*', '')
    try:
        dic ['year'] = int(pieces[1].strip())
    except:
        return dic
    
        
    dic ['journ'] = pieces[2].strip()
    if 'DOI' in pieces[-1]:
        dic ['DOI'] = pieces[-1].replace('DOI ', '').lower().strip(']').strip()
    vol_pieces = [ p for p in pieces if re.search('V\d', p)]
    if vol_pieces:
        dic ['vol'] = vol_pieces[0].replace('V', '').strip()
    page_pieces = [p for p in pieces if re.search('P\d', p)]
    if page_pieces:
        dic ['page'] = page_pieces[0].replace('P', '').strip()
    return dic

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
        ref.update({'citing_art': row['ID_num']})
    return refs




def make_rdf(df):
    '''Make a dataframe of all cited references in dataset.'''
    
    data= df[df[list(topics.keys())].sum(axis=1)>=1].dropna(subset=['CR'])
    
    
    return pd.DataFrame(
        utils.list_flattener(
        utils.parallelize_on_rows(
           data, ref_df_maker).tolist()
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
    #%%
def collect_cites(df, rdf):
    rdf['DOI'] = rdf["DOI"].apply(lower_if)
    
    df['DI'] = df['DI'].apply(lower_if)
    
    
    #perfect DOI matches:
    relevant = df.dropna(subset=['DI']).merge(rdf, left_on = 'DI', 
                right_on = "DOI", how = 'inner')[['DOI', 'DI', 'citing_art', 
                                                  #'J9', 
                                                  #'PY', 'journ', 
                                                  #'year', 'VL', 'vol',
                                                  #'page', 'BP'
                                                  ]]
                                    
    df.dropna(subset= ['AF'], inplace = True)
    df['ln_fi'] = df['AF'].apply(aut_ln_fi)
    rdf['ln_fi'] = rdf['aut'].apply(ref_aut_ln_fi)
    df['VL'] = df['VL'].apply(vol_formatter)
    
    
    
    #get matches where ln_fi of first author, journal name and year are all same
    m = df.merge(rdf, left_on = ['ln_fi', 'J9', 'PY'], right_on = ['ln_fi', 'journ', 
                 'year'], how ='inner')
    
    #subset these matches for where the citation volume or page number =
    #that of the possible article match, or both of these are missing. 
    m = m[((m['VL'] == m['vol']) | (m['vol'].isnull()))  
          & 
          ((m['BP'] == m['page']) | (m['page'].isnull()))
          ]
    
    m.loc[(m['DOI'].isnull()) & (m['DI'].isnull()==False), 'DOI'] = m['ID_num']
    refs = pd.concat([relevant, m[['DOI', 'DI', 'citing_art']]])
    return refs
   
#%%

if __name__=='__main__':
    path=os.path.join('data', 'all_data.csv')
    refs = collect_cites(df, make_rdf(df))
    refs.to_csv(os.path.join('data', 'ref_matrix.csv'))
    