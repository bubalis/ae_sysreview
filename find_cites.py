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
from utils import list_flattener, load_topics



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
        dic['DOI']=(re.sub('\[|\]|;', '', doi.search(ref).group()).strip())
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
    

testlist=[('Oksanen J.',
  '2011',
  'VEGAN COMMUNITY ECOL',
  'DOI 10.HTTP://CRAN.R-PROJECT.ORG/PACKAGE=VEGAN'),
 ('Oksanen J.',
  '2011',
  'VEGAN COMMUNITY ECOL',
  'DOI 10.HTTP://CRAN.R-PROJECT.ORG/PACKAGE=VEGAN'),
 ('Mishra P', '2014', 'IOSR J AGR VET SCI', 'DOI 10.9790/2380-07722326'),
 ('Mishra P', '2014', 'IOSR J AGR VET SCI', 'DOI 10.9790/2380-07722326'),
 ('이정동', '2014', 'DOI', 'DOI 10.9787/PBB.2014.2.4.334'),
 ('이정동', '2014', 'DOI', 'DOI 10.9787/PBB.2014.2.4.334'),
 ('Muller JW', '2016', 'WHAT IS POPULISM', 'DOI 10.9783/9780812293784'),
 ('Muller JW', '2016', 'WHAT IS POPULISM', 'DOI 10.9783/9780812293784'),
 ('Dendoncker N.',
  '2018',
  'SUSTAINABILITY SCI S',
  'DOI 10.9774/GLEAF.9781315620329_10'),
 ('Dendoncker N.',
  '2018',
  'SUSTAINABILITY SCI S',
  'DOI 10.9774/GLEAF.9781315620329_10'),
 ('Araya M.',
  '2006',
  'J CORPORATE CITIZENS',
  'DOI 10.9774/GLEAF.4700.2006.SP.00005'),
 ('Araya M.',
  '2006',
  'J CORPORATE CITIZENS',
  'DOI 10.9774/GLEAF.4700.2006.SP.00005'),
 ('Schaper M',
  '2002',
  'GREENER MANAGEMENT I',
  'DOI 10.9774/GLEAF.3062.2002.SU.00004')]
 
 

 
class sortedMatcher(object):
    '''Class for running sorted matches.
    Sorting by the match-field means 
    that each value needs to be checked only once.''' 
    
    
    def __init__(self, check_list, pos_matches):
        self.check_list=sorted(check_list)
        self.pos_matches=sorted(pos_matches)
        self.matches=[]
        self.non_matches=[]
        self.i=0
        
        
    def find_matches(self):
        ''' '''
        self.display()
        for item in self.check_list:
            while True:
                if not self.check_length():
                    return self.matches, self.non_matches
                result=self.match(item)
                if result=="Match":
                    self.matches.append(self.retrieve())
                    self.i+=1
                elif result=="Less than":
                    self.non_matches.append(self.retrieve())
                    self.i+=1
                elif result== "Greater than":
                    break
        return self.matches, self.non_matches
        
    def match_result(self, pos_match, item):
        if type(item)==str:
            return self.string_matchResults( pos_match, item)
        elif type(item)==tuple:
            return self.tup_matchResults(pos_match, item)
       
    def string_matchResults(self, pos_match, item):
        '''Matching function for strings.'''
        
        if not pos_match or pos_match==np.nan:
            return "Less than"
        if pos_match in item or item in pos_match: #partial matches are included
           return "Match"
        elif pos_match<item:
           return "Less than"
        elif pos_match>item:
           return "Greater than"
       
    def tup_matchResults(self, pos_match, item):
        '''Matching function for tuples.'''
        
        if not pos_match or pos_match==np.nan:
           return "Less than"
        if pos_match==item:
           return "Match"
        elif pos_match<item:
           return "Less than"
        elif pos_match>item:
           return "Greater than"
        
        
        
    def check_length(self):
        return self.i<len(self.check_list) and self.i<len(self.pos_matches)
    
    def display(self):
        print(self.check_list[20:30])
        print(self.pos_matches[20:30])
  
    #%%
class dfSortedMatcher(sortedMatcher):
    def __init__(self, check_list, pos_matches, sort_column):
        '''Args: List of references to check against.
        Possible matches. 
        Column for 
        '''
        self.check_list=sorted(check_list)
        self.pos_matches=pos_matches.sort_values(by=sort_column).reset_index(drop=True)
        self.sort_col=sort_column
        self.matches=[]
        self.non_matches=[]
        self.i=0
        self.df=self.pos_matches
        
    
    def match(self, item):
        pos_match=str(self.pos_matches[self.sort_col].iloc[self.i]).lower()
        return self.match_result(pos_match, item)
    
    def retrieve(self):
        return self.pos_matches.iloc[self.i]

    
    def get_matches(self, get_non_matches=True):
        matches, non_matches=self.find_matches()
        print(matches[0:10])
        print(self, '    ', len(matches))
        if get_non_matches:
            return pd.DataFrame(matches), pd.DataFrame(non_matches)
        else:
            return pd.DataFrame(matches)
   
    
class df_MatcherComplex(dfSortedMatcher):
    
        
    def match(self, item):
        check=self.pos_matches[self.i][0]
        return self.match_result(check, item)
    
    def retrieve(self):
        return self.pos_matches[self.i][1]
    
    def get_matches(self, get_non_matches=True):
        matches, non_matches=self.find_matches()
        matches=list(set(matches))
        non_matches=[i for i in range(len(self.df)) if i not in matches]
        if get_non_matches:
            return self.df.loc[matches], self.df.loc[non_matches]
        else:
            return self.df.loc[matches]
  
  
    
    
    
class dfMultiColumnMatcher(df_MatcherComplex):
    '''Matcher for information that is stored in multiple columns of a dataframe.'''
    def __init__(self, check_list, pos_matches):
        self.check_list=[self.check_formatter(item) for item in check_list if item]
        self.check_list=sorted([tup for tup in self.check_list if tup])
        self.pos_matches=[self.pos_matches_formatter(row, i) for i, row in pos_matches.iterrows()]
        self.pos_matches=sorted([tup for tup in self.pos_matches if tup])
        self.matches=[]
        self.non_matches=[]
        self.i=0
        self.df=pos_matches
        self.display()
        
class nonDOI_refmatcher(dfMultiColumnMatcher):
    '''Matcher for references, using information other than DOIs.'''   
    
    def check_formatter(self, item):
        '''Return tuple of author, year, item'''
        if all([item['Author'], item['Year'], item['Journal']]):
            return (item['Author'].lower(), item['Year'], item['Journal'])
    
    def pos_matches_formatter(self, row, i):
        '''Return Nested Tuple:
            tuple of author, year, item 
            and tuple of row index that it occurs at'''
        aut=self.author_formatter( row['AU'])
        yr=self.year_formatter(row['PY'])
        journ=row['J9']
        if all([aut, yr, journ]):
            return ((aut, yr, journ),i) 
    
    def author_formatter(self, aut_string):
        if aut_string and type(aut_string)==str:
            first_aut=aut_string.split(';')[0]
            return str(first_aut).strip().lower()
     
        
    def year_formatter(self, yr):
        yr=float(yr)
        if isnan(yr):
            return None
        else:
            return str(int(yr))
        
class dfSortMatchList(df_MatcherComplex):
    def __init__(self, check_list, pos_matches, column):
        dfSortedMatcher.__init__(self, check_list, pos_matches, column)
        self.df=pos_matches
        self.pos_matches=sorted(list_flattener([[(aut_cleaner(author).strip(), i) 
                        for author in string.split(';')] 
                        for i, string in pos_matches[column].iteritems() if 
                        type(string)==str]))
    
    def match_result(self, pos_match, item):
        if not pos_match or pos_match==np.nan:
            return "Less than"
        if pos_match.lower() in item.lower():
           return "Match"
        elif pos_match<item:
           return "Less than"
        elif pos_match>item:
           return "Greater than"
      
    
    

    
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
    refs_to_keep, not_refs=dfSortedMatcher(dois, possible_refs, 'DI').get_matches()
    more_refs_to_keep, not_refs=nonDOI_refmatcher(all_refs, not_refs).get_matches()
    
    out_df=to_keep.append(refs_to_keep)
    return out_df, not_refs

if __name__=='__main__':
    pass
    
    #out_df.to_csv('data\\articles_and_refs.csv')
    
    