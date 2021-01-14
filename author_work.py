# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:32:01 2019

@author: benja
"""


import pandas as pd
import re
import numpy as np
import os
from itertools import combinations, permutations
from multiprocessing import Pool
import utils
import string
import json


#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process

def aut_cleaner(string):
    '''CLean author names'''
    string=re.sub(r'"|\[|', '', string)
    string = re.sub(r"'|\]", '', string)
    parts = name_spliter(string)
    if len(parts)==2:
        if parts[1].upper() == parts[1] and len(parts[1]) <= 3 and '.' not in parts[1]:
            return string.replace(parts[1], '. '.join(parts[1])+'.')
    return string
               
    



def list_auts(paper_auts):
    '''Return all unique author names in string.'''
    autFind=re.compile(r'\[.+?\]')
    authors=autFind.findall(str(paper_auts))
    return [aut_cleaner(author).split(';') for author in authors]    



def list_flattener(nested_list):
    '''Flatten a nested list of any depth to a flat list'''
    while True:
        nested_list=[n for x in nested_list for n in x]
        if type(nested_list[0])!=list:
            return nested_list
        
def make_alist(series):
    '''Return a list of all authors, given a series representing the authors.'''    
    aut_nested_list=series.tolist()
    aut_nested_list= [a for a in aut_nested_list if type(a)==str]
    flat_list= list_flattener([section.split(';') for section in aut_nested_list])
    return list(set([item.strip() for item in flat_list]))


def titleCase_LN(names_string):
    '''Make all last names in in names string titlecase''' 
    for name in str(names_string).split(';'):
        names_string=names_string.replace(name.split(',')[0], name.split(',')[0].title())
    return names_string
    

def dict_find_replace(string, dic, split_char):
    '''Replace individual elements of a string with matches from a dictionary. 
    List function'''
    string=str(string)
    as_list=string.split(split_char) #make string into list
    for item in [item.strip() for item in as_list]: 
        replacement= dic.get(item, None)
        if replacement:
            string=string.replace(item, replacement)
    return string                



f_letter =re.compile(r'^.+?\,\s\w')
def initial_dic(a_list):
    '''Sort a list of authors into groups based on shared LN-First initial.
    args: list of names.
    returns dic of the form:
        {"Doe, J": 
         ["Doe, John", "Doe, John Q.", "Doe, J. Q.", "Doe, James"]}'''
    
    f_letter =re.compile(r'^.+?\,\s\w')
    dic={} 
    a_list.sort()
    match_list=[]
    for author in a_list:
        if not match_list:
            if f_letter.search(author):
                #ln_fi
                ln_fi=f_letter.search(author).group()
                match_list.append(author)
        else:
            if ln_fi in author:
                match_list.append(author)
            else:
                key=ln_fi+'.'
                dic[key]=match_list
                match_list=[]
    return dic



def uniq(li):
    return list(set(li))
#%%
def not_ln(name):
    '''Return all Elements of name that are not the last name.'''
    return name.split(',')[1]

def initial_matches(name_list):
    '''Make intial round of matches for a names list.'''
    
    init_dict=initial_dic(name_list)
    length=len(init_dict)
    out={}
    for i, k in enumerate(init_dict):
        v=init_dict[k]
        if i%100==0 and i!=0:
            print(f'{(i/length)*100}% completed')
        if len(v)>1:
            out[k]=subDict(v, name_list).run()
    return out
    
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

def checkname_parts(list_a, list_b):
    return all([(n.strip('.') in [i[0] for i in list_b]) or (f'{n[0]}.' in list_b) 
                or (n in list_b) 
                 for n in list_a])

def check_all_initials_or_names(name1, name2):
    name1_as_list=[a for a in name_spliter(name1) if a]
    name2_as_list=[a for a in name_spliter(name2) if a]
    min_pieces=min([len(name1_as_list), len(name2_as_list)])
    print (name1_as_list, name2_as_list, min_pieces)
    return all([name1_as_list[i][0]==name2_as_list[i][0] for i in range(min_pieces)])
           

def len_name_components(name):
    '''Return number of name parts a name has. '''
    return len(name_spliter(name))

    
def match_initials(name1, name2):
    '''Return whether all intials of two names  are the same letter.'''
    return all([p1[0]==p2[0] for p1, p2 in zip(name1.parts, name2.parts)])


def match_full_names(name1, name2):
    '''Return that no full name parts are mismatched.'''
    zip_obj=zip(name1.part_codes, 
                    name2.part_codes,
                    name1.parts, 
                    name2.parts)
    
    return all([p1==p2 for c1, c2, p1, p2 in 
               zip_obj if 
              isinstance(c1, Full) and
              isinstance(c2, Full)])

def match_initials_longer(name1, name2):
    '''If name1 is shorter than name2, return whether all initials are the same letter.
    Otherwise return False.'''
    
    if len(name1.parts)<=len(name2.parts):
        return match_initials(name1, name2)
    return False


def length_max(li,  exclude_char='.'):
    '''Return longest string in list, excluding exclude_char'''
    if li:
        li=list(li)
        li.sort()
        li.sort(key=lambda x: len(x.replace(exclude_char, '')))
        
        return li[-1]





def retrieve_part_code(string):
    if len(string)==2 and string[-1]=='.':
        return InitialwDot()
    elif all([len(string)>2, 
              '-' in string, 
              string==string.upper()]):
        return InitialsWDash()
    elif len(string)>2 and '-' in string:
        return FullwDash()
    elif len(string)>1 and string==string.upper():
        return ListOfInitials()
    elif len(string)==1:
        return InitialNoDot()
    elif len(string)>2 and '-' in string:
        return FullwDash()
    elif string:
        return Full()
    else:
        return None

class NamePartCode(object):
    def __init__(self):
        pass

class ListOfInitials(NamePartCode):

    def __repr__(self):
        return 'List of Initials'

class InitialwDot(NamePartCode):
    
    def __repr__(self):
        return 'Initial With a Dot'    

class FullwDash(NamePartCode):
    def _repr__(self):
        return 'Full Name With Dash'

class InitialNoDot(NamePartCode):

    def __repr__(self):
        return 'Initial Without a Dot' 

    
class InitialsWDash(NamePartCode):
    
    def __repr__(self):
        return 'Initial with a Dash'


class Full(NamePartCode):


    def __repr__(self):
        return 'Full String' 



def match_parts(name1, name2, stop, start=0):
    return all([p1==p2 for p1, p2 in 
                zip(name1.parts[start:stop], name2.parts[start:stop])])



def return_if_unique(func):
    def wrapper(func):
        out_list=func()
        if len(set(out_list))==1:
            return out_list[0]
        else:
            return None
    return wrapper
        
def return_spec_max(func):
    def wrapper(func):
        out_list=func()
        return length_max(out_list)
    return wrapper

class UserDecorators(object):
    @classmethod
    def return_if_unique(cls, func):
        def wrapper(cls):
            out_list=func(cls)
            if len(set(out_list))<=2:
                return length_max(out_list)
            else:
                return None
        return wrapper
    
    @classmethod
    def return_spec_max(cls, func):
        def wrapper(cls):
            out_list=func(cls)
            return length_max(out_list)
        return wrapper

class Name(object):
    def __init__(self, string, subDict):
        self.string=string
        self.subDict=subDict
        self.out_name=string
        self.clean()
        self.update(string)
        
        self.standardize_initials()
        
    def __repr__(self):
        return f'Name obj: {self.string}'
        
    def update(self, string):
        self.parts=name_spliter(string)
        self.part_codes=[retrieve_part_code(part) for part in self.parts if part]
        #if self.out_name!=string:
            #print(string)
        self.out_name=string
    
    
    def make_step(self,method):
        f=self.__getattribute__(method)
        res=f()
        if res:
            self.update(self.retrieve_parent(res))
    
    
    
    
    def retrieve_parent(self, string):
        if string!=self.out_name and string in self.subDict.names:
            parent=self.subDict.names[string]
            return parent.retrieve_parent(parent.out_name)
        else:
            return string
        
    def name_w_dash(self):
        
        pass
    
    @UserDecorators.return_if_unique
    def two_init(self):
        '''Finds two-initial matches and reassigns values.'''
        
        matches=[]
        for k, v in self.subDict.names.items():
            if match_initials_longer(self, v):
                matches.append(k)
        return matches
    
    
    def standardize_initials(self):
        new_string=self.parts[0]+', '
        for part, code in zip(self.parts[1:],self.part_codes[1:]):
            if isinstance(code, ListOfInitials):
                part=''.join([f'{p}. ' for p in list(part)])
            elif isinstance(code, InitialNoDot):
                part=part+'.'
            elif isinstance(code, InitialsWDash):
                part=''.join([f'{p}. ' for p in part.split('-')])
            new_string+=part+' '
        self.update(new_string.strip())
            
    def clean(self):
        if "'" in self.out_name or '"' in self.out_name:
          self.out_name= self.out_name.replace("'", '').replace('"', '')
    
    #@check_run   
                     
    
    #@check_run
    @UserDecorators.return_if_unique
    def fn_match(self):
        '''Return if first name matches'''
        string=self.out_name
        match_list=[]
        if '.' not in string.split()[1]:
            for k, v in self.subDict.names.items():
                if (match_parts(self, v, start=0, stop=2) and 
                match_initials((k, string))): 
                    
                    match_list.append(k)
                
        return match_list
                    
    
    #@check_run 
    @UserDecorators.return_if_unique
    def full_string(self): 
        '''Function searches for all full-string matches with key in subdic  and reassigns values 
        to the longest version'''
        match_list=[]
        for k, v in self.subDict.names.items():
            if self.out_name in k and not isinstance(self.part_codes[-1],InitialwDot):
                #print(self.out_name)
                match_list.append(k)
            
        return match_list 

    #@check_run    
    def one_init(self): ##Function searches for unique First-initial Lastname matches with key in a lower-level self.dicctionary 
        '''Function searches for unique FI, 
        LN matches with key in the lower-level dictionary and reassigns values '''
        if len(set(self.subDict.names.values()))==2:
            for key in [key for key in list(self.subDict.names) if re.search(r'\w+\,\s\w\.?$', key) and (self.subDict.names[key]==key)]:
                v=length_max(self.subDict.names.values())
                if key[:-1] in v:
                   return [v]
                    
    #@check_run
    #def update(self):
     #   '''Reassigns values to the values of the keys that they point to, if longer than current value'''
      #  string=self.out_name
       # for k, v in self.subDict.names.items():
        #    if self.subDict.names[string]==k:
                

    #@check_run
    @UserDecorators.return_spec_max
    def dash (self):
        '''Deal with random dashes by checking if they can be eliminated or replaced with space'''
        string=self.out_name
        
        if '-' in string:
            out_list=[]
            for k in self.subDict.names:
                if any([string.replace('-', ' ' ).lower() in k.lower(),
                        string.replace('-', '' ).lower() in k.lower(),
                        all([p.lower() in k.lower() for p in string.split('-')])]):
                    out_list.append(k)
            return out_list
     
    def two_init_final(self):
        pass
            

def same_struct(name1, name2):
    return ((len(name1.parts)==len(name2.parts)) 
            and all([type(p1)==type(p2) for p1, p2 in 
                     zip(name1.part_codes,
                         name2.part_codes)]))

def no_inconsistencies(name1, name2):
    if name1.out_name==name2.out_name:
        return True
    return all([not same_struct(name1, name2),
                match_full_names(name1, name2),
                match_initials(name1, name2)])
            


class subDict(object):
    '''Take a list of names that share  Lastname, First initial and match them using various methods.
    Can exlude some functions to make matching less aggressive'''
    

    def __init__(self, sub_list, full_list, exclude=[], depth=1):
        self.names={name:Name(name, self) for name in sub_list}
        self.flags=exclude
        self.full_list=full_list
        #self.dict={n:n for n in sub_list}
        self.steps=['full_string', 
                    'dash', 
                    'fn_match', 
                    'two_init', 
                    'dash',
                'one_init', 
                           'two_init_final', 
                           'one_init']
        self.steps=[s for s in self.steps if s not in exclude]
        self.depth=depth
    
    def all_different_structs(self):
        combos=combinations(self.names.values(),2)
        if all([no_inconsistencies(name1, name2) for name1, name2 in combos]):
            out_string=length_max([n.out_name for n in self.names.values()])
            for name in self.names.values():
                name.update(out_string)
    
    def recursive_2_initials(self):
        
        three_part_names={k: v for k,v in self.names.items() if len(v.parts)>2}
        secnd_inits=[name.parts[2][0] for name in three_part_names.values()]
        for init in [i for i in secnd_inits if i]:
            li=[k for k, v in three_part_names.items() if v.parts[2][0]==init]
            d=subDict(li, li, exclude=['one_init', 'fn_match'], depth=2)
            matches=d.run()
            for k,v in{k:v for k,v in matches.items() if k!=v }.items():
                self.names[k].update(v)
            
    def make_step(self, method):
        if len(set([name.out_name for name in self.names.values()]))>1:
            for name in self.names.values():
                name.make_step(method)
                
       
    def run(self):
        if self.depth==1:
            self.recursive_2_initials()
        
        self.all_different_structs()
        #print({k: v.out_name for k,v in self.names.items()})
        for method in self.steps:
           self.make_step(method)
           if len(set([name.out_name for name in self.names.values()]))>1:
               break
        self.all_different_structs()
        return {k:v.out_name for k, v in self.names.items()}
    
#%%

    
    

        

 
def condition_checker(statement):
    def evaluator():
        locals().update(globals())
        return eval(statement)
    return evaluator
    
def solved_to_point(nested_dict):
    '''Return all subdicts that are "solved": have only one out_val.'''
    dictionary={}
    for v in {k:v for k,v in nested_dict.items()}.values():
        dictionary={**dictionary, **{key: value for key,value in v.items() if key!=value}}
    return dictionary

def not_solved_yet(nested_dict, solved_dict, name_list):
    '''Return a dictionary of all sub_dicts that are still ambiguous, 
    have at least one name in one or two initial form, 
    and have at least one value in the solve dictionary.'''
    return {k:v for k,v in nested_dict.items() 
            if all([len(set(v.values()))>1,
            any([p in solved_dict.values() for p in v.values()]),
            k in v.values(),
            re.search(r'[^\s]+\,\s\w\.(\s\w\.|$)', '$'.join(v.values()))])} 


def make_matches(df, aut_col, author_inst_col, other_cols=[]):
    df[aut_col]=df[aut_col].astype(str).apply(titleCase_LN)
    a_list=make_alist(df[aut_col])
    dict_1=initial_matches(a_list)
    matches_1=solved_to_point(dict_1)
    df[aut_col]=[dict_find_replace(au, matches_1, split_char=';') for au in df[aut_col].to_list()]
    return matches_1, df,


def unpack_nested_di(nested_di):
    out={}
    for di in nested_di.values():
        out.update(di)
    return out

def dict_find_replace(string, dic, split_char):
    '''Replace individual elements of a string with matches from a dictionary. 
    List function'''
    string=str(string)
    as_list=string.split(split_char) #make string into list
    for item in [item.strip() for item in as_list]: 
        replacement= dic.get(item, None)
        if replacement:
            string=string.replace(item, replacement)
    return string               


def main(data_path, save_path):
    print('Matching Authors')
    df=pd.read_csv(data_path)    
    df['AF']=df['AF'].astype(str).apply(titleCase_LN)
    a_list=make_alist(df['AF'])
    dict_1=initial_matches(a_list)
    full_matches=solved_to_point(dict_1)
    any_matches=unpack_nested_di(dict_1)
    df['AF']=df['AF'].apply(lambda x: dict_find_replace(x, any_matches, ';'))
    df.to_csv(save_path, index=False)
    return full_matches, a_list, dict_1


def test_main(data_path, n=10000):
    df=pd.read_csv(data_path).head(n)
    df['AF']=df['AF'].astype(str).apply(titleCase_LN)
    a_list=make_alist(df['AF'])
    dict_1=initial_matches(a_list)
    matches=solved_to_point(dict_1)
    return matches, a_list, dict_1

#%%
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
    return [aut_cleaner(name).strip() for name in string.split(';')]

def parse_names2(string):
    pieces=[p.strip() for p in string.split(';')]
    return [aut_cleaner(p.split('/')[0]).strip() for p in pieces]
    
name_cols=['AU', 'AF']
id_cols=['RI', 'OI']   

def collect_all_names(row):
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
         OI: ORCID ID if any'''   
    
    
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
    
def remove_dupe_tupes(li_of_tupes):
    for line in sorted(li_of_tupes, key=len):
        if any([
                all([i in o_line for i in line]) 
                for o_line in  
                [o for o in li_of_tupes if o!=line]
                ]):
            li_of_tupes.remove(line)
    return li_of_tupes


def collect_extra_names(df, id_col):
    '''Get Extra Names for a numeric identifier.'''
    sub=df[df[id_col].isnull()==False]
    data=sub.groupby(id_col)['names'].apply(flatten_and_remove_dupes)
    return data

def flat_w_uniq(nested_li):
    return list(set(utils.list_flattener(nested_li)))


def flat_uniq_tup_df(nested):
    if nested.any():
        if type(nested) in (tuple, list):
            return tuple(flat_w_uniq(nested))
    
    
#%%

def get_other_names(df1, col, other, other_col, both):
    '''Get the other names for an identifier number by checking its matching
    identifier from the other identifier set.'''
    
    
    merg = df1.merge(both, left_index=True, right_on=col, how='left')
    merg = merg[merg.index.isnull()==False]
    merg = merg.merge(other, left_on=other_col, right_on=other_col, how='left')
    merg['names'] = (merg['names_x']+merg['names_y']).apply(lambda x: list(set(x)))
    return merg[[col, 'names']]
    
def make_aut_df(df):
    '''Get all name-forms associated with each numeric identifier for the two
    identifier lists.'''
    
    df['article_index'] = df.index
    adf = pd.DataFrame(utils.list_flattener(
        utils.parallelize_on_rows(
            df, all_ids_row).to_list()
        )
        )
    
    
    return adf

def retrieve_identifiers(adf):
    RIs = identifier_df(adf, 'RI', 'OI')
    OIs = identifier_df(adf, 'OI', 'RI')
    both = adf[(adf['RI'].isnull()==False) & (adf['OI'].isnull()==False)]
    both = pd.DataFrame(both.groupby(['RI', 'OI'])['names'].apply(flat_w_uniq)).reset_index()
    both.drop(columns='names', inplace=True)
    RIs = get_other_names(RIs, 'RI', OIs, 'OI', both)
    OIs = get_other_names(OIs, 'OI', RIs, 'RI', both)
    return RIs, OIs


def flat_uniq_nonnan(data):
    return [i for i in flat_w_uniq(list(data)) if i]

def ln_fi_from_group(li):
    return ln_fi_get(sorted(li, key=len)[-1])

def max_len(li):
    return max(li, key=len)


def is_full_name(name):
    return count_letters(name.split(',')[-1])>2


def identifier_df(adf, ID, other_id):
    gb = adf.groupby(ID)
    df = gb[['names', 'inst', 'email', 'postal_code', 'indicies']].agg(flat_uniq_nonnan)
    df[other_id]=gb[other_id].agg('first')
    for column in ['inst', 'email', 'postal_code']:
        df.loc[df[column].isnull(), column] = [[]] * df[column].isnull().sum()
        assert np.nan not in df[column].tolist()
    with Pool(8) as pool:   
        df['longer_name']=pool.map(max_len, df['names'])
        df['ln_fi']=pool.map(ln_fi_get, df['longer_name'])
        
    gb = df.reset_index().groupby('longer_name')
    df = gb[['names', 'inst', 'email', 'postal_code', 'indicies']].agg('first')
    df['longer_names'] = df['names'].apply(
        lambda x: [i for i in x if count_letters(i.split(',')[-1])>2])
    for column in ['inst', 'email', 'postal_code']:
        df.loc[df[column].isnull(), column] = [[]] * df[column].isnull().sum()
    df[ID] = gb[ID].agg('first')
    df[other_id] = gb[other_id].agg('first')
    df['longer_name'] = gb['longer_name'].agg('first')
    df['ln_fi'] = gb['ln_fi'].agg('first')
    df.set_index(ID, inplace=True)
    
    
    return df.rename(columns={'names': 'aut_tuple'})
     

def count_letters(stri):
    return len(re.findall('|'.join(string.ascii_letters), stri))

def break_into_sub_cols(df, col):
    try:
        col_names=[f'{col}_{x}' for x in range(df[col].apply(len).max())]
    
        df[col_names]=df[col].apply(pd.Series)
        return df
    except:
        print(col)
        raise

def collect_extra_names(li_of_di, id_col):
    pass


def tuple_match(tup1, tup2):
    return all([t in tup2 for t in tup1])

def get_keys(tup_of_names, keys):
    return [k for k in keys if tuple_match(tup_of_names, k)]

f_letter_name=re.compile(r'^.+?\,\s\w$')
def authorfix(name):
    pieces=name.split(',')
    if len (pieces)==2:
        name=pieces[0].title()+',' +pieces[1]
    if f_letter_name.search(name):
        return name+'.'
    return name
    
def aut_col_cleaner(names):
    if type(names)==str:
        return '; '.join([authorfix(name.strip()) for name in names.split(';')])
    else:
        return names
#%%

        
#%%
def code_df_as_dict(df):
    '''Convert the author-code df to a dictionary.'''
    return {tuple(item): num for 
            item, num in zip(df['aut_tuple'].tolist(), df.index.tolist())}


def retrieve_matches(names, code_df):
    '''Get a list of matches for author_tuples. 
    Return the author ID num for all tuples that have:
    2 different name-forms and 
    one unique matching ID num that matches both name-forms.
    args: names: list of name_tuples,
    code_df: a df of author codes.
    col_name: name of the author id number code. 
    Returns: dictionary.
    '''
    di = code_df_as_dict(code_df)
    matches = match_all_para([n for n in names if n[0]!=n[1]], list(di.keys()))
    matches = {k:v[0] for k, v in matches.items() if len(v)==1}
    num_matches = {k : di.get(v) for k, v in matches.items()}
    return num_matches 

def rebuild_df(series_list_dics):
    '''Last step in converting a paralellized split operation back to a dataframe'''
    
    return pd.DataFrame(
        utils.list_flattener(series_list_dics.tolist()))


def matcher(names, keys):
    matches={name:get_keys(name, keys) for name in names}
    return {k:v for k, v in matches.items() if v}

def match_all_para(names, keys):
    splits=zip([[n for n in names if n[0][0]==fl] for fl in string.ascii_uppercase],
               [[key for key in keys if key[0][0]==fl] for fl in string.ascii_uppercase])
    with Pool() as pool:
        matches=pool.starmap(matcher, splits)
    
    
    return {k: v for d in matches for k, v in d.items()}
    



def partial_matches(names, keys, fl):
    matches={}
    sub_names=[n for n in names if n[0][0]==fl]
    sub_keys=[key for key in keys if key[0][0]==fl]
    for name in sub_names:
        match=get_keys(name, sub_keys)
        if match:
            matches[name]=match
    return matches



def ln_fi_get(name):
    if f_letter.search(str(name)):
        return f_letter.search(name).group()


def get_email(row):
    emails=row['EM']
    if type(emails)!=str:
        return None
    emails=emails.split(';')
    try:
        string= re.search('\w{3,5}', row['longer_name'].split(',')[0].lower()).group()
    except:
        return None
    
    for em in emails:
        if string in em:
            return em.strip()

def extract_solved(adf):
    gb1 = adf.groupby('longer_name').count()
    gb1=gb1.reset_index()
    gb1['ln_fi']=gb1['AF'].apply(ln_fi_get)
    gb2 = gb1.groupby('ln_fi').count()
    solved = gb2[gb2['aut_tuple']==1].index    
    return solved

#%%
def match_by(adf, col_names):
    
    gb = adf.dropna(subset=col_names).groupby(
        ['ln_fi']+ col_names)
    matches = pd.DataFrame(gb['aut_tuple'].agg(flat_uniq_nonnan))
    matches['indicies'] = gb['indicies'].agg(list)
    matches['longer_name'] = gb['longer_name'].agg(max_len)
    #matches['ln_fi'] = gb['ln_fi'].agg('first')
    matches=matches.reset_index()
    matches['id_num'] = matches.index.map(lambda x: '_'.join(col_names)+str(x))
    return matches
            






def match_from_dfs(df1, df2, col='aut_tup'):
    di = {item[-2]: item[-1] for item in df2.to_records()}
    li = df1[col].tolist()
    matches = matcher(li, di.keys())
    matches = {k: di.get(v) for k, v in matches.items()}
    df1['id_num']=df1['id_num'].apply(
        lambda row: matches.get(row[col]) if np.isnan(row['id_num']) else row['id_num'],
        axis=1)
    
    return df1
        


def split_by_first_letter(df, col):
    '''Split a dataframe into a list of 26 dataframes.
    Each one contains values for col where the first letter of that string is
    that letter.'''
    return [df[df[col].apply(lambda x: str(x)[0])==letter] 
                 for letter in string.ascii_uppercase]

def add_IDs(adf, matchers):
    '''Add ID numbers from various matchers.'''
    
    for m in matchers:
        data_splits=split_by_first_letter(df, 'ln_fi')
        match_splits=split_by_first_letter(m, 'ln_fi')
        with Pool(8) as pool:
            adf=pd.concat(pool.starmap(
                match_from_dfs, zip(data_splits, match_splits)))
    return adf   


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
    '''For parsing the corresponding author address in column "RP" '''
    split_string=r'\),'
    if name in string:
        return string.split(split_string)[1]

def get_address(row):
    '''Extract the author address from either column C1 (inst affiliation)
    or from column RP (corresponding author).'''
    
    string=row['C1']
    name=row['AF']
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




postal_codes = json.loads(open(os.path.join('data', 'postal-codes.json')).read())
postal_codes = [item for item in postal_codes if item['Regex'] and item['ISO']]
pos_codes = [(item['Country'], item['ISO'], item['Regex'][1:-1].replace(r'\b', '')) for item in postal_codes]



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
        for item in pos_codes:
            try:
                pattern=re.compile(item[1]+'?(,|$)')
                if item[0] in string or pattern.search(string):
                    if re.search(item[2], string):
                        return item[1]+ ' ' + re.search(item[2], string).group()
            except:
                print(item)
                raise
                
    return None

def extract_inst(string):
    if type(string)==str:
        return string.split(',')[0]
    else:
        return None





#%%



def combine_matchers(ids, other, col_name):
    ids=break_into_sub_cols(ids, col_name)
    for sub_col in [c for c in ids.columns if f'{col_name}_' in c]:
        ids=glom_matcher(ids, other, col_name, sub_col)
        ids.drop(columns=[sub_col], inplace=True)
    return ids


def glom_matcher(ids, other, col_name, sub_col):
    ind_name =ids.index.name
    m = ids.reset_index().merge(other, 
            left_on = ['ln_fi', sub_col], 
            right_on = ['ln_fi', col_name], 
            how = 'inner').set_index(ind_name)
    
    m['aut_tuple'] = (m['aut_tuple_x'] + m['aut_tuple_y']).apply(uniq)
    m['indicies'] = (m['indicies_x'] + m['indicies_y']).apply(uniq)
    indicies_to_keep = (m['aut_tuple'].apply(len)>m['aut_tuple_x'].apply(len))
    print(f'{indicies_to_keep.sum()} new matches found')
    m = m[indicies_to_keep]
    ids.loc[m.index, 'aut_tuple'] = m['aut_tuple']
    ids.loc[m.index, 'indicies'] = m['indicies']
    return ids





#all_as['OI'] = all_as['aut_tuple'].map(retrieve_matches(names, OIs,'OI'))
#all_as['ln_fi']=all_as['AF'].apply(ln_fi_get)
#to_solve = all_as[(all_as['RI'].isnull()) & (all_as['OI'].isnull())]

def make_uniq_names(adf):
    val_counts = pd.DataFrame(adf.value_counts('ln_fi')).reset_index()
    uniq_names = val_counts[val_counts[0]==1]['ln_fi']
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
    m = adf[adf[id_col].isnull()].merge(
        ids.reset_index(), 
    left_on = left_on, 
    right_on = right_on, 
    how = 'inner')
    if not m.empty:
        old_matches=(adf[id_col].isnull()==False).sum()
        col_names =[c for c in ['email', 'inst', 'postal_code'] if c not in left_on]
        for col_name in ['email', 'inst', 'postal_code']:
            ids.loc[m[f'{id_col}_y'], f'{col_name}'] = m.apply(lambda row: add_to_list_df(row, col_name), axis=1).tolist()
        
        for col in col_names: 
            ids[col] = nan_to_emptylist(ids[col])
            
        print(f'{m.shape[0]} Matches Found')
        adf.loc[m['aut_index'], id_col] = m[f'{id_col}_y'].tolist()
        new_matches=(adf[id_col].isnull()==False).sum()
        print (new_matches - old_matches)
        
    return adf, ids


def multi_val_matcher(adf, ids, id_col, col_name):
    ids=break_into_sub_cols(ids, col_name)
    for sub_col in [c for c in ids.columns if f'{col_name}_' in c]:
        adf, ids = match_by_columns(adf, ids, id_col, 
                                    left_on = ['ln_fi', col_name],
                                    right_on = ['ln_fi', sub_col])
        
        ids.drop(columns=[sub_col], inplace=True)
    return adf, ids


def long_name_matcher(adf, ids, id_col):
    ids=break_into_sub_cols(ids, 'longer_names')
    for sub_col in [c for c in ids.columns if 'longer_names_' in c]:
        adf, ids = match_by_columns(adf, ids, id_col, 
                                    left_on = ['longer_name'],
                                    right_on = [sub_col])
        ids.drop(columns=[sub_col], inplace=True)
    return adf, ids

def all_numeric_matchers(adf, ids, id_col):
    adf, ids = long_name_matcher(adf, ids, id_col)
    for other_id in ['email', 'inst', 'postal_code']:
        adf, ids= multi_val_matcher(adf, ids, id_col, other_id)
    return adf, ids

#%%
data_path=os.path.join('data', 'all_data.csv')
df=pd.read_csv(data_path)


df['AU'] = df['AU'].apply(aut_col_cleaner) 
df['AF'] = df["AF"].apply(aut_col_cleaner)

#%%
columns_to_use =  ['AF', "AU", "EM", "OI", "RI", "C1", "RP", ]

adf = make_aut_df(df.loc[:, columns_to_use])
df=None

#%%

print('Formatting Author Data')
adf['address'] = utils.parallelize_on_rows(adf, get_address) 
with Pool(8) as pool:   
    adf['ln_fi']=pool.map(ln_fi_get, adf['AF'])
    adf['postal_code'] = pool.map(get_postal_code, adf['address'])
   

adf['aut_index'] = adf.index


#adf.to_csv(os.path.join('data', 'big_author_temp.csv'))

#adf.drop(index=uniq_names.index, inplace=True)

adf['aut_tuple'] = adf[['AF', 'AU']].to_records().tolist()
adf['longer_name'] = adf['aut_tuple'].apply(lambda x: x[-1])    

adf['email']=utils.parallelize_on_rows(adf, get_email)
adf['inst']=adf['address'].apply(extract_inst)

print("Matching by Email, Postal Code and Institution")


#%%
print('Making Identifier Dataframes')
RIs = identifier_df(adf, 'RI', 'OI')
assert np.nan not in RIs['email'].tolist()
OIs = identifier_df(adf, 'OI', 'RI')
assert np.nan not in OIs['email'].tolist()

for ids, col in zip((RIs, OIs), ('RI', 'OI')):
    adf[col].loc[adf[col].isin(ids.index) == False] = np.nan


adf, RIs = all_numeric_matchers(adf, RIs, 'RI')
adf, OIs = all_numeric_matchers(adf, OIs, 'OI')

#%%
def first_valid(row, columns):
    val = None
    for col in columns:
        val = row[col]
        if val:
            return val



uniq_names=make_uniq_names(adf)
to_solve = adf.loc[(adf['RI'].isnull()) & (adf['OI'].isnull()) & (adf['ln_fi'].isin(uniq_names) ==False )]


def alternate_matcher(adf, col_name):
    '''Function for finding matches based on other column values. '''
    matches = match_by (adf, [col_name])
    adf[f'{col_name}_id'] = None 

    m = adf.merge(
        matches, 
    left_on = col_name, 
    right_on = col_name, 
    how = 'inner')
    
    if not m.empty:
        adf.loc[m['aut_index'], f'{col_name}_id'] = m['id_num'].tolist()
    m = adf.merge(
        matches[matches['longer_name'].apply(is_full_name)], 
    left_on = 'longer_name', 
    right_on = 'longer_name', 
    how = 'inner')
    if not m.empty:
        adf.loc[m['aut_index'], f'{col_name}_id'] = m['id_num'].tolist()
    
    return adf, matches

matcher_dict={}
for col_name in ['email', 'postal_code', 'inst']:
    adf, matcher_dict[col_name] = alternate_matcher(adf, col_name)
    assert f'{col_name}_id' in adf.columns
    

match_cols= ['OI', "RI", 
                'email_id', 
                'inst_id', 
                'postal_code_id']


left= adf[(adf[match_cols].fillna(False).astype(bool).sum(axis =1) == 0) 
          & (adf['ln_fi'].isin(uniq_names) ==False)]

codes = adf[(adf[match_cols].fillna(False).astype(bool).sum(axis =1) != 0)][['OI', "RI", 
                'email_id', 
                'inst_id', 
                'postal_code_id', 'longer_name', 'ln_fi']]

m = left.merge(codes, left_on='ln_fi', right_on = 'ln_fi', how = 'inner')
m['matches'] = m.apply(
    lambda row: all_parts_match(row['longer_name_x'], row['longer_name_y']), 
    axis=1)

for col in match_cols:
    adf.loc[m[m['matches']]['aut_index'], col] = m[m['matches']][f'{col}_y'].tolist()

gb=left.dropna(subset=['ln_fi']).groupby('ln_fi')

indexer = pd.DataFrame(gb['longer_name'].agg(
                    list
                    ).apply(
                        lambda li: all([all_parts_match(x,y) 
                                        for x,y in permutations(li,2)])
                        ))
                        

indexer = indexer.merge(left, on = 'ln_fi', how = 'inner')
indexer = indexer.loc[indexer['longer_name_x']]
indexer['name_index'] = indexer.reset_index()['index'].apply(lambda x: f'name{str(x)}').tolist()

adf.loc[indexer['aut_index'], 'name_index'] = indexer['name_index'].tolist()

if 'name_index' not in adf.columns:
    adf['name_index'] = None

    match_cols.append('name_index')                   

left= adf[(adf[match_cols].fillna(False).astype(bool).sum(axis =1) == 0) & (adf['ln_fi'].isin(uniq_names) ==False)]

left['name_index'] = left.reset_index().index.to_numpy()+adf.shape[0]
left['name_index']= left['name_index'].apply(lambda x: f'name{str(x)}')
adf.loc[left['aut_index'], 'name_index'] = left['name_index'].tolist()


adf.to_csv('all_author_data.csv')


'''
vs=RIs['email'].value_counts()

print('Matching to emails, postal codes and institutions')
for matchers, column in zip((em_matches, zip_matches, inst_matches), 
                            ('email', 'postal_code', 'inst' )):
    
    RIs = combine_matchers(RIs, matchers, column)
    OIs = combine_matchers(OIs, matchers, column)



indexer=(adf['OI'].isnull()) & (adf['RI'].isnull())
names=adf[indexer]['aut_tuple'].tolist()

print('Finding Name Matches')
adf.loc[adf['OI'].isnull(), 'OI'] = adf[adf['OI'].isnull()]['aut_tuple'].map(retrieve_matches(names, RIs,))
                                        

indexer=(adf['OI'].isnull()) & (adf['RI'].isnull())

adf.loc[adf['RI'].isnull(), 'RI'] = adf[indexer]['aut_tuple'].map(retrieve_matches(names, RIs))

new_id_cols=['email_id', 'postal_id', 'inst_id']
indexer=(adf['OI'].isnull()) & (adf['RI'].isnull())

for match_df, col_name in zip([em_matches,  zip_matches, inst_matches], ['email_id', 'postal_id', 'inst_id']):
    indexer=adf[new_id_cols+['OI', 'RI']].isnull().sum(axis=1)==5
    names=adf[indexer]['aut_tuple'].tolist()
    print(f'Assigning {col_name}')
    adf.loc[indexer, col_name] = adf[indexer]['aut_tuple'].map(retrieve_matches(names, match_df.set_index('id_num')))
    

#%%
indexer=adf[new_id_cols+['OI', 'RI']].isnull().sum(axis=1)==5
to_solve=adf[indexer]
to_solve=to_solve[to_solve['ln_fi'].isin(uniq_names.tolist())==False]

counts = to_solve['ln_fi'].value_counts()
'''
'''Left to do: take remaining authors and match based on longest name.'''



#%%
#
#if __name__=='__main__':
    #data_path=os.path.join('data', 'all_data.csv')
    #df=pd.read_csv(data_path) 
    #RIs, OIs=make_aut_df(df)   
    #all_as=pd.DataFrame(utils.list_flattener(
    #    utils.parallelize_on_rows(df.dropna(subset=['AF']), utils.split_row).tolist()))
    
    
    #'''
    #save_path=os.path.join('data', 'corrected_authors.csv')
    #matches, alist, dict1=main(data_path, save_path)
    #with open(os.path.join('data','author_list.txt'), 'w+') as f:
    #    print('\n'.join(alist), file=f)
    #'''
