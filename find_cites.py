# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:17:21 2019

@author: benja
"""


import os
import numpy as np
import pandas as pd
import re
from author_work import aut_cleaner
from math import isnan
from utils import list_flattener, load_topics, load_col_names



findjourn=re.compile('[A-Z\s]{4,}')
doi=re.compile('DOI\s10.+$')
findyear=re.compile(r'(?<=\,\s)\d\d\d\d')
find_aut=re.compile('.+?(?=\,\s\d\d\d\d)')
topics=load_topics()

def year_parser(ref):
    '''Find a year in a string.'''
    
    if findyear.findall(ref):
        for num in findyear.findall(ref):
            if int(num) in range(1300, 2020): #Sanity Check of Year. 
                return num
    else:
        return None


def parse_ref(ref):
    '''Turn reference string into dictionary:
        Author, year, Journal, DOI'''
    dic={}
    dic['Author']=ref.split(',')[0].strip()  ###Return First Item as the author
    dic['Year']=year_parser(ref)
    if findjourn.search(ref):
       dic['Journal']=(findjourn.search(ref).group().strip())
    else:
        dic['Journal']=None
    if doi.search(ref):
        dic['DOI']=doi.search(ref).group().strip()
        #dic['DOI']=(re.sub('\[|\]|;', '', doi.search(ref).group()).strip())
    else:
        dic['DOI']=None
    return dic





def remove_dupes(ref_list, dupe_index):
    '''Initial method for Removing Duplicate citations.
    dupe_index: the index of the component that may be a duplicate.
    Returns: List of unique references in their longest forms.
    List of duplicate references'''
    longest_refs=[]
    dupelist=[]
    ref_list.sort(key= lambda x: x[dupe_index])
    i=0
    while i<len(ref_list):
        item=ref_list[i]
        checklist=[item]
        i+=1
        if item[dupe_index]:
            while True:
                check= ref_list[i]
                if check_matches(item, ref_list[i], dupe_index):
                    checklist.append(check)
                    i+=1
                else:
                    break
                
        else:
            pass  
        if len(checklist)==1:
            longest_refs.append(item)
        else:
            longest_refs.append(max(checklist, key= lambda x: len(' '.join(x))))
            dupelist.append(checklist)

    return longest_refs, dupelist


def check_matches(cite1, cite2, dupe_index):
    '''Check two citations are the same.'''
    if len(cite2)<=dupe_index:
        return False
    return cite1[dupe_index]==cite2[dupe_index]



def str_concat(*names, join_char):
    return join_char.join(names)


def space_concat(*names):
    return str_concat(*names, join_char=' ')

def row_space_concat(row):
    return space_concat(*row.to_list())

vect_length=np.vectorize(len)



def format_ref_df(df, columns, aut_dict):
    for column in columns:
        df[f'orig_{column}']=df[column]
    df['Author']=df['Author'].map(aut_dict)
    for col in [c for c in columns if c!='Author']:
        pass
    

def check_all_matches(df, ref_cols):
    df['total_ref_length']=vect_length(df.apply(row_space_concat, axis=1))
    to_keep, dupes=unique_DOIs(df)

def ind_of_unique_vals(df, columns):
    '''Return index values of df where column has unique values.'''
    indicies=df.groupby(columns)['total_ref_length'].idxmax()
    return indicies




def unique_DOIs(df, column, dupes=pd.DataFrame()):
    indicies=ind_of_unique_vals(df, [column])
    to_keep=df.loc[indicies]
    dupes.append(df.loc[[i for i in range(df.shape[0]) if i not in indicies]])
    return to_keep, dupes
    
 
 

 

    
    

    
def collect_refs(df, ref_col):
    '''List all refs in the reference column of the dataframe'''
    all_refs=list_flattener([f.split(';') 
            for f in df[ref_col].tolist() 
            if type(f)==str])
    return [parse_ref(ref) for ref in all_refs]

#%%




def main(path):
    df=pd.read_csv(path)
    print('Finding Citation Matches')
    to_keep=df[df[list(topics.keys())].sum(axis=1)>=1]
    
    print('Shape of dataFrame:    ', to_keep.shape)
    possible_refs=df[df[list(topics.keys())].sum(axis=1)==0]
    all_refs=collect_refs(to_keep, 'CR')
    dois=[ref['DOI'].replace('DOI ', '').lower() for ref in all_refs if type(ref['DOI'])==str]
    dois=list(set(dois))
    
    refs=possible_refs[possible_refs['DI'].str.lower().isin(dois)]
    not_refs=possible_refs.loc[[d for d in possible_refs.index if d not in refs.index]]
    first_aut=lambda x: str(x).split(';')[0].replace(',', '')
    not_refs['author_yr_pub']=(not_refs['AU'].apply(first_aut)+
                               ', '+not_refs['PY'].fillna('-1').astype(int).astype(str)+
                               ', '+not_refs['J9']).str.lower()
    
    ref_strs=[f'{ref["Author"]}, {ref["Year"]}, {ref["Journal"]}'.lower() for ref in all_refs]
    more_refs=not_refs[not_refs['author_yr_pub'].isin(ref_strs)]
    out_df=to_keep.append(refs).append(more_refs)
    return out_df, refs, more_refs

if __name__=='__main__':
    path=os.path.join('data', 'corrected_authors.csv')
    out_df, refs, more_refs=main(path)
    out_df.to_csv(os.path.join('data', 'articles_and_refs.csv'))
    
    