# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:32:01 2019

@author: benja
"""


import pandas as pd
import re
import numpy as np
import os
from itertools import permutations
import multiprocessing as mp
import utils
import string
import json
import sys
from functools import partial


def aut_cleaner(string):
    '''CLean author names'''
    string=re.sub(r'"|\[|', '', string)
    string = re.sub(r"'|\]", '', string)
    
    return initials_list_to_initials(string)

def initials_list_to_initials(string):
    '''Convert a name Doe, JF to Doe, J. F.'''
    parts = name_spliter(string)
    if len(parts)==2:
        if parts[1].upper() == parts[1] and len(parts[1]) <= 3 and '.' not in parts[1]:
            return string.replace(parts[1], '. '.join(parts[1])+'.')
    
    return string
           
f_letter =re.compile(r'^.+?\,\s\w')
#load in postal codes data
postal_codes = json.loads(open(os.path.join('data', 'postal-codes.json')).read())
postal_codes = [item for item in postal_codes if item['Regex'] and item['ISO']]
pos_codes = [
    {'country': item['Country'], 
              'iso': item['ISO'], 
              'regex': item['Regex'][1:-1].replace(r'\b', '')} 
             for item in postal_codes]





def uniq(li):
    return list(set(li))

def only_letters(stri):
    return ''.join([i for i in stri if i in string.ascii_letters])

def all_parts_match(name1, name2):
    return all([
    all([c1 == c2 for c1, c2 in 
         zip(only_letters(p1), only_letters(p2))]) 
    for p1, p2 in zip(name1.split(), name2.split())])


def name_spliter(name):
    '''Split name into its component parts.'''
    name_spliter_re=re.compile(r'\,|\s')
    try:
        return [part.strip() for part in name_spliter_re.split(name) if part.strip()]
    except:
        print(name)

    
def name_num_dict(item):
    '''Turn an entry for ORCID IDs or Researcher IDs into a dict of 
    Name: id_num'''
    pieces=[p for p in item.split(';') if p.strip()]
    try:
        vals=[piece.split('/') for piece in pieces]
        return {name.strip(): num.strip() for name, num in [v for v in vals if len(v)==2]}
    except:
        print(pieces)
        raise
    
def consolidate_name_num_di(li_of_di):
    '''Parse a list of dictionaries to make one name_num_dict'''
    
    out={}
    errors=[]
    for di in li_of_di:
        #if all values are either missing from dict or the same:
        if all([out.get(key, value)==value for key, value in di.items()]):
            out.update(di)
        else:
            errors+list(di.keys())
    return out, errors


def name_num_dict_all(df, col):
    '''Turn a column with ORCID or Researcher ID numbers into a dictionary of 
    Name: Id_num pairs.'''
    valid=df.dropna(subset=[col])
    return consolidate_name_num_di(
        valid[col].apply(name_num_dict)
        )
#%%    
def parse_names1(string):  
    '''Name Parser for AU/AF columns.'''
    return [aut_cleaner(name).strip() for name in string.split(';')]

def parse_names2(string):
    '''Name Parser for OI/RI columns.'''
    pieces=[p.strip() for p in string.split(';')]
    return [aut_cleaner(p.split('/')[0]).strip() for p in pieces]
    
name_cols=['AU', 'AF']
id_cols=['RI', 'OI']   

def collect_all_names(row):
    '''Get all versions of author name from a row of the dataframe.
    Return all unique names, and a dictionary of id numbers.'''
    
    names=[]
    dis={}
    for col in name_cols:
        if type(row[col])==str:
            names+=parse_names1(row[col])
            
    for col in id_cols:
        if type(row[col])==str:
            names+=parse_names2(row[col])
            dis[col]=name_num_dict(row[col])
        else:
            dis[col]={}
            
    return list(set(names)), dis

def make_name_dict(names, di, index):
    '''From  a list of names and a nested dictionary of
    ID_nums, return a list of dics: 
        {names: list of associated names,
         RI: Researcher ID if any.
         OI: ORCID ID if any}'''   

    
    out=[]
    fls=list(set([f_letter.search(n).group() for n in names if 
                  f_letter.search(n)]))
    
    for fl in fls:
        fl_names=[n for n in names if fl in n]
        if len (fl_names) >= 2:
            au = fl_names[0]
            af = fl_names[1]
        else:
            au = None
            af = None 
        codes={}
        for key, sub_dict in di.items():
            for n in fl_names:
                if n in sub_dict:
                    codes[key]=sub_dict[n]
                    break
                elif len (n.split(', '))==2:
                    n=n.split(', ')[1]+', '+n.split(', ')[0]
                    if n in sub_dict:
                        fl_names.append(n)
                        codes[key]=sub_dict[n]
            else:
                codes[key]=None
        
        if au:
            sub_out={**codes, **{'names': fl_names, 'indicies': index, 
                             'AU': au, 'AF': af} } 
        
            out.append(sub_out) 
    return out


def all_ids_row(row):
    '''Collect all author identifiers for a given article. 
    Returns a list of dicts''' 
    names, dis=collect_all_names(row)
    return [{**{col: row[col] for col in row.index}, **row_dict} 
        for row_dict in make_name_dict(names, dis, row.name)]
    

def flat_w_uniq(nested_li):
    return list(set(utils.list_flattener(nested_li)))

#%%
    
def make_aut_df(df):
    '''Get all name-forms associated with each numeric identifier for the two
    identifier lists.'''
    df['index'] = df.index
    adf = pd.DataFrame(utils.list_flattener(
        utils.parallelize_on_rows(
            df, all_ids_row, num_of_processes=6).to_list()
        )
        )    
    
    adf.rename(columns= {'names': 'aut_tuple', 'index': 'ID_num'},
               inplace = True)
    return adf

def flat_uniq_nonnan(data):
    return [i for i in flat_w_uniq(list(data)) if utils.is_valid(i)]

def ln_fi_from_group(li):
    return ln_fi_get(sorted(li, key=len)[-1])

def max_len(li):
    return max(li, key=len)


def is_full_name(name):
    '''Return whether the second part of a name has at least 2 letters in it.'''
    if ',' in name:
        return count_letters(name.split(',')[-1])>2
    return False



def full_name_matches(li):
    '''From a list of names, return a dictionary of possible matches. '''
    
    matches = {}
    for i in sorted(li, key= len, reverse = True):
        res = []
        for i2 in li:
            if all_parts_match(i, i2) and (count_letters(i)>=count_letters(i2)) and i != i2:
                res.append(i2)
        matches[i] = res
        
    return unpack_di_list(matches)


def unpack_di_list(di):
    '''Unpack a dictionary where the values are lists.
    Assign unique matches where all elements. ''' 
    di = {k:v for k, v in di.items() if v}
    di2 = {}
    flat = utils.list_flattener(list(di.values()))
    
    for k, li in sorted(di.items(), key = lambda x: len(x[0])): 
        if all([all_parts_match(n1, n2) for n1, n2 in permutations(li,2)] ):
            for name in li:
                
                #Assign only if match is unique. 
                if name not in di.keys() or k not in flat:
                    
                    #delete if item is already there
                    if di2.get(name):
                        di2[name]= 'Null'
                    
                    else:   
                        di2[name] = k
                        
    return {k:v for k,v in di2.items() if v != 'Null' }
#%%

def identifier_df(adf, ID, other_id):
    '''Make a  dataframe for the OId or RId column.'''
    
    gb = adf.groupby(ID)
    df = gb[['aut_tuple', 'inst', 'email', 'postal_code', 'indicies']].agg(flat_uniq_nonnan)
    df[other_id]=gb[other_id].agg('first')
    
    for column in ['inst', 'email', 'postal_code']:
        #assign empty lists for null values
        df.loc[df[column].isnull(), column] = [[]] * df[column].isnull().sum()
        
        
    with mp.Pool(mp.cpu_count()) as pool:   
        df['longer_name'] = pool.map(max_len, df['aut_tuple'])
        df['last_name_first_init'] = pool.map(ln_fi_get, df['longer_name'])
        
    gb = df.reset_index().groupby('longer_name')
    df = gb[['aut_tuple', 'inst', 
             'email', 'postal_code', 'indicies']].agg('first')
    
    #collect all 'long versions' of name
    df['longer_names'] = df['aut_tuple'].apply(
        lambda x: [i for i in x if count_letters(i.split(',')[-1])>2])
    
    for column in ['inst', 'email', 'postal_code']:
        df.loc[df[column].isnull(), column] = [[]] * df[column].isnull().sum()
     
    cols = [ID, other_id, 'longer_name', 'last_name_first_init']
    
    df[cols] = gb[cols].agg('first')

    df.set_index(ID, inplace=True)
    
    
    return df
     

def count_letters(stri):
    '''Return number of letters in a string.'''
    return len(re.findall('|'.join(string.ascii_letters), stri))



def break_into_sub_cols(df, col):
    '''Take a column that has a list of values in it,  and exploded it such that each value
    has its own column. 
    e.g. : df = 
                   x
            0  [1, 1, 1]
            1   [ 2, 2]
                 
    break_into_sub_cols(df, 'x') yields:
        
                       x          x_0  x_1  x_2
                   0  [1, 1, 1]    1    1    1
                   1     [2, 2]    2    2    NaN
    '''
  
    col_names=[f'{col}_{x}' for x in range(df[col].apply(len).max())]

    df[col_names]=df[col].apply(pd.Series)
    return df
   



def tuple_match(tup1, tup2):
    return all([t in tup2 for t in tup1])

def get_keys(tup_of_names, keys):
    return [k for k in keys if tuple_match(tup_of_names, k)]

f_letter_name=re.compile(r'^.+?\,\s\w$')
def authorfix(name):
    '''Make three fixes to name '''
    
    return   list_of_initials_fix(
        f_initial_no_dot_fix(
        titlecase_fix(
            name)
        )
        )
   

def titlecase_fix(name):
    '''Ensure that last name is titlecase '''
    pieces=name.split(',')
    if len (pieces)==2:
        return pieces[0].title()+',' +pieces[1]
    else:
        return name

def f_initial_no_dot_fix(name):
     '''Turn name of form Doe, J to Doe, J.'''
     if f_letter_name.search(name):
        return name+'.'
     return name

def list_of_initials_fix(string):
    '''Turn a name of the form 'Johnson, LB'  to
    "Johnson, L. B. for consistency." '''
    
    parts = name_spliter(string)
    if len(parts)==2:
        if parts[1].upper() == parts[1] and len(parts[1]) <= 3 and '.' not in parts[1]:
            return string.replace(parts[1], '. '.join(parts[1])+'.')
    return string

def aut_col_cleaner(names):
    '''Clean author strings, then put them back together.''' 
    if type(names)==str:
        return '; '.join([authorfix(name.strip()).lower() for name in names.split(';')])
    else:
        return names

#%%
def code_df_as_dict(df):
    '''Convert the author-code df to a dictionary.'''
    return {tuple(item): num for 
            item, num in zip(df['aut_tuple'].tolist(), df.index.tolist())}


def rebuild_df(series_list_dics):
    '''Last step in converting a paralellized split operation back to a dataframe'''
    
    return pd.DataFrame(
        utils.list_flattener(series_list_dics.tolist()))


def ln_fi_get(name):
    '''Get the last-name first-initial form of the name.
    e.g. ln_fi_get(Doe, John) 
    return Doe, J.'''
    
    if f_letter.search(str(name)):
        return f_letter.search(name).group()


def get_email(row):
    '''If the EM column in the df has an email that matches
    the author's first 3 to 5 letters of last name and first initial,  
    return that email as a string. 
    '''
    
    emails=row['EM']
    if type(emails)!=str:
        return None
    emails=emails.split(';')
    try:
        string1 = re.search('\w{3,5}', row['longer_name'].split(',')[0].lower()).group()
        string2 = row['last_name_first_init'][-1]
    except:
        return None
    
    for em in emails:
        if '@' not in em:
            continue
        n = em.split('@')[0]
        if string1 in n and string2 in n:
            return em.strip()




#%%
def make_matcher(adf, col_names):
    '''Create a df for an alternate match field (or set off fields).
    Return a dataframe with columns:
        [col_names} + [indicies, aut_tuple, longer name and an
                       id number absed on those columns.'''
    gb = adf.dropna(subset=col_names).groupby(
        ['last_name_first_init']+ col_names)
    
    matches = pd.DataFrame(gb['aut_tuple'].agg(flat_uniq_nonnan))
    matches['indicies'] = gb['indicies'].agg(list)
    matches['longer_name'] = gb['longer_name'].agg(max_len)
    #matches['last_name_first_init'] = gb['last_name_first_init'].agg('first')
    
    matches=matches.reset_index()
    
    matches['id_num'] = matches.index.map(lambda x: '_'.join(col_names)+str(x))
    
    return matches
            


def split_by_first_letter(df, col):
    '''Split a dataframe into a list of 26 dataframes.
    Each one contains values for col where the first letter of that string is
    that letter.'''
    return [df[df[col].apply(lambda x: str(x)[0])==letter] 
                 for letter in string.ascii_uppercase]

name_group=re.compile('\[.*?\]')

def parse_multi_address(string):
    '''Turn a multi-address "[names] address [names] more address" 
    Into a dictionary of names and their associated addresses.'''
    names=re.findall(name_group, string)
    names=[parse_str_list(n) for n in names]
    addresses=[i.strip() for i in re.split(name_group, string) if i]
    out={}
    for name_list, address in zip(names, addresses):
        for name in name_list:
            out[name]=address
    return out

def parse_address_2(string, name):
    '''For parsing the corresponding author address in column "RP" 
    Split on the string 'corresponding author' Name appears before this string,
    address appears after it. 
    '''
    
    pieces = re.split('\(corresponding author\),', string)
    for i, piece in enumerate(pieces):
        if name in piece:
           return pieces[i+1].split(';')[0].strip()

def get_address(row):
    '''Extract the author address from either column C1 (inst affiliation)
    or from column RP (corresponding author).'''
    
    string=row['C1']
    name=row['AU']
    if type(string)==str:
        if '[' in string:
            return parse_multi_address(string).get(name)
        else:
            return string
    elif type(row['RP'])==string:
        return parse_address_2(row["RP"], name)
    
    else:
        return None


def parse_str_list(string, sep=';'):
    '''Return a string that encodes a list as a list.'''
    string = string.replace('[', '').replace(']', '')
    return [i.strip() for i in string.split(sep)]




def get_postal_code(string):
    '''Get the postal code, if any encoded in the address. 
    Uses a list of postal code regexes. Returns:
        a string of the format: "{ISO_country_code} {postal code}"
        e.g. "US 01450"'''
        
    if type(string) != str:
        return None
    
    if 'US' in string and re.search(r'\d{5}', string):
        return 'US'+' ' + re.search(r'\d{5}', string).group()
    
    else:
        for item in pos_codes: #iterate through dict of postal codes
            if check_postal_code_string(string, item):
                 return item['iso']+ ' ' + re.search(item['regex'], string).group()
            
    return None

def check_postal_code_string(string, item):
    '''Check whether a string matches a national postal code format.
    Must have: right format and the country's name or iso abbrv '''
    
    iso_re=re.compile(item['iso']+'?(,|$)')
    if item['country'] in string or iso_re.search(string): #check for country name or ISO abbrv
        return bool(re.search(item['regex'], string))
           

def extract_inst(string):
    '''Extract institution from an address string.'''
    if type(string)==str:
        return string.split(',')[0]
    else:
        return None


#%%
def make_uniq_names(adf):
    val_counts = pd.DataFrame(adf.value_counts('last_name_first_init')).reset_index()
    uniq_names = val_counts[val_counts[0]==1]['last_name_first_init']
    return uniq_names


def add_to_list_df(row, col_name):
    try:
        if not row[f'{col_name}_x']:
            return row[f'{col_name}_y']
        elif row[f'{col_name}_x'] in row[f'{col_name}_y']:
            return row[f'{col_name}_y']
        else:
            return list(set([row[f'{col_name}_x']]+row[f'{col_name}_y']))
    except:
        print(col_name)
        raise


def nan_to_emptylist(data):
    data.loc[data.isnull()] = [[]] * data.isnull().sum()
    return data

def match_by_columns(adf, ids, id_col, left_on, right_on):
    '''Find addtional matches by matching on other columns.
    Find any name entries that don't have an RI/OI, 
    but have an email/inst/postal_code (with last-name first-initial) 
    that matches a RI or OI entry.
    Return the adf, updated with the id column filled in 
    and the ids updated with the new matches, 
    encoded as a list of indicies of matching entries.'''
    
    m = adf[adf[id_col].isnull()].merge(
        ids.dropna(subset=right_on).reset_index(), 
    left_on = left_on, 
    right_on = right_on, 
    how = 'inner')
    
    if right_on[-1] =='email_0':
        m.to_csv(os.path.join('data', f'{id_col}_emails.csv'))
    try:
        if not m.empty:
            
            col_names =[c for c in ['email', 'inst', 'postal_code'] if c not in left_on]
            
            #update the ids data_frame
            for col_name in ['email', 'inst', 'postal_code']:
                ids.loc[m[f'{id_col}_y'], f'{col_name}'] = m.apply(
                    lambda row: add_to_list_df(row, col_name), axis=1).tolist()
            for col in col_names: 
                ids[col] = nan_to_emptylist(ids[col])
                
            print(f'{m.shape[0]} Matches Found')
           
            #update author dataframe
            adf.loc[m['aut_index'], id_col] = m[f'{id_col}_y'].tolist()
    except:
        print(m.columns, ids.columns, adf.columns)
        raise
        
        
    return adf, ids


def multi_val_matcher(adf, ids, id_col, col_name):
    '''Add id numbers to the author dataframe by matching on another 
    column identifier: email, institution or postal code.'''
    
    ids=break_into_sub_cols(ids, col_name)
    
    for sub_col in [c for c in ids.columns if f'{col_name}_' in c]:
        
        adf, ids = match_by_columns(adf, ids, id_col, 
                                    left_on = ['last_name_first_init', col_name],
                                    right_on = ['last_name_first_init', sub_col])
        
        ids.drop(columns=[sub_col], inplace=True)
        
    return adf, ids


def long_name_matcher(adf, ids, id_col):
    '''Add IDs to the adf from an id_dataframe based 
    on exact matches of long-form names.'''
    
    ids['longer_names'] = ids['longer_names'].apply(lambda li: 
                                            [i for i in li if is_full_name(i)])
        
    ids=break_into_sub_cols(ids, 'longer_names')
    print('Matching by longer name and  ', id_col)
    for sub_col in [c for c in ids.columns if 'longer_names_' in c]:
        
        adf, ids = match_by_columns(adf, ids, id_col, 
                                    left_on = ['longer_name'],
                                    right_on = [sub_col])
        
        ids.drop(columns=[sub_col], inplace=True)
    return adf, ids

def all_numeric_matchers(adf, ids, id_col):
    '''Perform all matches to find pre-provided id numbers (RIs and OIs).
    adf : author dataframe
    ids : the identifier dataframe
    id_col : the identifier column
    
    each matching function collects additional data from other matching columns,
    e.g. if the ids df has an id# for Doe, John and finds another Doe, John,
    that whole entry becomes linked; the email address, institution and zip code
    all are added to the adf. 
    
    
    '''
    
    #Add to the 
    adf, ids = long_name_matcher(adf, ids, id_col)
    for other_id in ['email', 'inst', 'postal_code']:
        print(f'Matching by {id_col} and {other_id}')
        adf, ids= multi_val_matcher(adf, ids, id_col, other_id)
    return adf, ids

#%%

        
#%%


def alternate_matcher(adf, col_name):
    '''Function for finding matches based on other column values. '''
    matches = make_matcher(adf, [col_name])
    adf[f'{col_name}_id'] = None 

    m = adf.merge(
        matches, 
    left_on = col_name, 
    right_on = col_name, 
    
    how = 'inner')
    
    if not m.empty:
        print(f'{m.shape[0]} matches found')
        print(m[col_name].head(5))
        adf.loc[m['aut_index'], f'{col_name}_id'] = m['id_num'].tolist()
    m = adf.merge(
        matches[matches['longer_name'].apply(is_full_name)], 
    left_on = 'longer_name', 
    right_on = 'longer_name', 
    how = 'inner')
    if not m.empty:
        adf.loc[m['aut_index'], f'{col_name}_id'] = m['id_num'].tolist()
    
    return adf, matches

def try_merge(left, right, **kwargs):
    try:
        return left.merge(right, **kwargs)
    
    except MemoryError:
        out = []
        right.to_csv('scratch.csv')
        reader = pd.read_csv('scratch.csv', chunksize = 1000)
        for r in reader:
            out.append(left.merge(r, **kwargs))
            del r
        return pd.concat(out)


def row_names_match(row):
    return all_parts_match(row['longer_name_x'], row['longer_name_y'])

def match_to_codes_on_full_name(adf, match_cols, uniq_names):
    '''Find any remaining full-name matches. 
    First  between remaining names and '''
    left= adf[(adf[match_cols].fillna(False).sum(axis =1) == 0) 
              & (adf['last_name_first_init'].isin(uniq_names) ==False)].copy()
    
    codes = adf[(adf[match_cols].fillna(False).sum(axis =1) != 0)][['OI', "RI", 
                    'email_id', 
                    'inst_id', 
                    'postal_code_id', 'longer_name', 'last_name_first_init']].copy()
    
    codes = codes.groupby('longer_name').agg('first')
    
    try: 
        #assign identifier info rows with full-name matches to the code df:
        m = try_merge(left, codes, 
                      left_on='last_name_first_init', 
                      right_on = 'last_name_first_init', how = 'inner')
        
        del codes
        if not m.empty:
            m['matches'] = m.apply(row_names_match, axis=1)
            
            
            for col in match_cols:
                adf.loc[m[m['matches']]['aut_index'], col] = m[m['matches']][f'{col}_y'].tolist()
            
        
            name_matches = all_fn_li_match(left)
            adf['name_id'] = None
            adf.loc[name_matches['aut_index'], 'name_id'] = name_matches['name_id'].tolist()
        return adf
    except:
        globals().update(locals())
        raise

def all_fn_li_match(left):
    '''Search for First-name last-initial groups where all names are consistent.
    Assign these a 'name ID' string. 
    '''
    
    gb=left.dropna(subset=['last_name_first_init']).groupby('last_name_first_init')
    
    indexer = pd.DataFrame(gb['longer_name'].agg(
                        list
                        ).apply(
                            lambda li: all([all_parts_match(x,y) 
                                            for x,y in permutations(li,2)])
                            ))
    indexer['name_id'] = indexer.reset_index().reset_index()['index'].apply(
        lambda x: f'name{str(x)}').tolist()
                       
    indexer = indexer.merge(left, left_on = 'last_name_first_init', right_on = 'last_name_first_init', 
                            how = 'inner')
    indexer = indexer.loc[indexer['longer_name_x']]
    
    return indexer

def li_o_di_to_di(li_o_di):
    '''Convert a list of dicts to one dict.'''
    out_dict={}
    for di in li_o_di:
        out_dict.update(di)
    return out_dict


def assign_remaining_names(adf, match_cols):
    '''Final Pass through names that do not yet have matches.
    Assign all of them a unique identifier. Then check to see if any of them 
    match with one another, purely based on name parts. '''
    
    #Give a unique identifier to each unique value for longer-name
    left = adf[adf[match_cols].fillna(False).astype(bool).sum(axis =1) == 0].copy()
    
    left_uniq = pd.DataFrame(
        left['longer_name'].unique(), columns = ['longer_name']).reset_index()
    
    left = left.merge(left_uniq, left_on = 'longer_name', 
            right_on ='longer_name', how ='left')
    left['name_id'] = left['index'].apply(lambda x: f'name_{str(x + adf.shape[0])}')
    
    
    #get all longer-name values for each lastname, first-initial value
    names = left.groupby('last_name_first_init')['longer_name'].agg(list).apply(uniq)
    
    #Dict of unique matches from a full-name matcher
    out_dict = li_o_di_to_di(
        names.apply(
            full_name_matches).tolist())

    
    left['match_name'] = left['longer_name'].apply(lambda x: out_dict.get(x,x))
    
    #recombine data and retreive new name_ids:
    keep_cols = ['match_name', 'longer_name', 'aut_index']
    matcher = left[keep_cols].merge(left[['name_id', 'longer_name']], 
                         right_on = 'longer_name', 
                         left_on = 'match_name', how ='inner')
    
    matcher = matcher[matcher['longer_name_x'] != matcher['longer_name_y']]
    left= left.merge(matcher, left_on ='aut_index', 
                     right_on = 'aut_index',  how = 'left')
    
    left['name_id'] = left[['name_id_y', 'name_id_x']].apply(
                                                    utils.first_valid, axis = 1)
    
    #assign name_ids numbers
    adf.loc[left['aut_index'], 'name_id'] = left['name_id'].tolist()

    return adf

def assign_name_string(adf):
    '''Assign the longest Name string available 
    to each unique author identifier. 
    Return the adf with that name in the column labelled "id_name"'''
    gb = adf.groupby('author_id')['longer_name'].agg(lambda x: max(x, key=len))
    
    adf = adf.merge(gb, left_on = 'author_id', 
                    right_on = 'author_id', how = 'left')
    
    return adf.rename(columns = {'longer_name_x' : 'longer_name', 
                                 'longer_name_y': 'id_name'})


def clean_data(df):
    df['AU'] = df['AU'].apply(aut_col_cleaner) 
    df['AF'] = df["AF"].apply(aut_col_cleaner)
    for col in ["EM", "OI", "RI"]:
        df[col] = df[col].apply(lambda x: str(x).lower())
    return df


def prep_aut_df(adf):
    '''Add columns to the author dataframe: 
        aut_index, address, email, inst, last_name_first_initial and postal code'''
    adf['address'] = utils.parallelize_on_rows(adf, get_address) 
    
    with mp.Pool(mp.cpu_count()) as pool:   
        adf['last_name_first_init']=pool.map(ln_fi_get, adf['AF'])
        adf['postal_code'] = pool.map(get_postal_code, adf['address'])
       
    
    adf['aut_index'] = adf.index
    assert adf['ID_num'].isnull().sum()== 0
    
    
    adf['longer_name'] = adf['aut_tuple'].apply(lambda x : sorted(x[1:], key = len)[-1])    
    
    adf['email']=utils.parallelize_on_rows(adf, get_email)
    adf['inst']=adf['address'].apply(extract_inst)
    
    return adf



if __name__ == '__main__':
    test = False
    if test:
        test_string = '_test'
    else:
        test_string = ''
    
    if len (sys.argv) == 1:
        save_dir = os.path.join('data', 'intermed_data')
        data_path=os.path.join(save_dir,  'all_data_2.csv')
        save_path=os.path.join(save_dir, f'all_author_data{test_string}.csv')
        article_data_save_path = os.path.join(save_dir,
                                              f'expanded_authors{test_string}.csv')
    elif len(sys.argv) == 4:
        data_path = sys.argv[1]
        save_path = sys.argv[2]
        article_data_save_path =sys.argv[3]
        
    else:
        raise ValueError(f'Script needs 3 arguments, data path and save path. {len(sys.argv)} args provided')
    
    df=pd.read_csv(data_path)
    if test:
        df = df.sample(20000)
    print(df.shape)
    #assert df['ID_num'].isnull().sum()==0
    
    df = clean_data(df)
    
    columns_to_use =  ['AF', "AU", "EM", "OI", "RI", "C1", "RP" ]
    
    adf = make_aut_df(df.loc[:, columns_to_use])
    print(f'Working with {adf.shape[0]} author-entries.')
    assert adf['ID_num'].isnull().sum()==0
    df=None
    
    print('Formatting Author Data')
    adf = prep_aut_df(adf)
    
    print("Matching by Email, Postal Code and Institution")
    
    print('Making Identifier Dataframes')
    
    RIs = identifier_df(adf, 'RI', 'OI')
    assert np.nan not in RIs['email'].tolist()
    OIs = identifier_df(adf, 'OI', 'RI')
    assert np.nan not in OIs['email'].tolist()
    
    for ids, col in zip((RIs, OIs), ('RI', 'OI')):
        adf[col].loc[adf[col].isin(ids.index) == False] = np.nan
    
    print('Matching by RI')
    adf, RIs = all_numeric_matchers(adf, RIs, 'RI')
    print('Matching by OI')
    adf, OIs = all_numeric_matchers(adf, OIs, 'OI')
    assert adf['ID_num'].isnull().sum() == 0
    
   
    uniq_names=make_uniq_names(adf)
    to_solve = adf.loc[(adf['RI'].isnull()) & (adf['OI'].isnull()) & (adf['last_name_first_init'].isin(uniq_names) ==False )]
    
    assert adf['ID_num'].isnull().sum()==0
    matcher_dict={}
    for col_name in ['email', 'postal_code', 'inst']:
        print(f'Using {col_name} for matches')
        adf, matcher_dict[col_name] = alternate_matcher(adf, col_name)
        assert f'{col_name}_id' in adf.columns
        
    assert adf['ID_num'].isnull().sum()==0
    match_cols= ['OI', "RI", 
                    'email_id', 
                    'inst_id', 
                    'postal_code_id']
    
    #RIs.to_csv(os.path.join('data',f'RIs{test_string}.csv')) #can 
    #OIs.to_csv(os.path.join('data', f'OIs{test_string}.csv'))
    del OIs
    del RIs
    
    adf = match_to_codes_on_full_name(adf, match_cols, uniq_names)
    match_cols.append('name_id')  
    assert adf['ID_num'].isnull().sum()==0
    adf.to_csv(save_path)
    
    print('Finding pure name-based matches.')
    adf = assign_remaining_names(adf, match_cols)
    
    adf['author_id'] = adf[match_cols].apply(utils.first_valid, axis = 1)
    
    adf = assign_name_string(adf)
    
    assert adf['ID_num'].isnull().sum()==0
    del to_solve
    del ids
    
    adf = adf[['ID_num', 'longer_name', 'author_id']]
    print('Saving Data')
    adf.to_csv(save_path)
    df = pd.read_csv(data_path)
    adf=adf.merge(df, left_on = 'ID_num', 
             right_on= 'index', how = 'left')
    
    
    adf.to_csv(article_data_save_path)
    


