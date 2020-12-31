# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:32:01 2019

@author: benja
"""


import pandas as pd
import re
import numpy as np
import os
from itertools import combinations

#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process

def aut_cleaner(string):
    '''CLean author names'''
    string=re.sub(r'"|\[|', '', string)
    return re.sub(r"'|\]", '', string)
    
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
    elif len(string)>1 and string==string.upper():
        return ListOfInitials()
    elif len(string)==1:
        return InitialNoDot()
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

class InitialNoDot(NamePartCode):

    def __repr__(self):
        return 'Initial Without a Dot' 

    
class InitialWDash(NamePartCode):
    
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
        name_strings=[name for name in self.subDict.names]
        if string!=self.out_name and string in self.subDict.names:
            parent=self.subDict.names[string]
            return parent.retrieve_parent(parent.out_name)
        else:
            return string
        
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
            elif isinstance(code, InitialWDash):
                part=''.join([f'{p}. ' for p in part.split('-')])
            new_string+=part+' '
        self.update(new_string.strip())
            
    def clean(self):
        if "'" in self.out_name or '"' in self.out_name:
          self.out_name= self.out_name.replace("'", '').replace('"', '')
    
    #@check_run   
                     
    def two_init_no_dot(self):
        pass
          
    
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
    def dash (self):
        '''Deal with random dashes by checking if they can be eliminated or replaced with space'''
        string=self.out_name
        if '-' in string:
            for k in self.subDict.names:
                if any([string.replace('-', ' ' ).lower() in k.lower(),
                        string.replace('-', '' ).lower() in k.lower(),
                        all([p.lower() in k.lower() for p in string.split('-')])]):
                    return [k]
     
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

    
    
class TestSubdict(subDict):
    def __init__(self, sub_list, full_list, exclude=[], fail_conditions={}):
        subDict.__init__(self, sub_list, full_list, exclude=[])
        self.fail_conditions=fail_conditions
        
        
    def asserter(self, step):
        for k,v in self.fail_conditions.items():
            assert self.names[k].out_name!=v, f'Failed on {step}, condition: {k}={v}'

    def match(self):
        if len(list(set(self.dict.values())))>1:
            for step in self.steps:
                #print(step.__name__)
                if self.check_run(step):
                    step()
                    self.asserter(step)
        
        return self.dict
    
    def make_step(self, method):
        subDict.make_step(self, method)
        self.asserter(method)
        

 
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





def test_set(li, fail_conditions={}):
    d=TestSubdict(li, li, fail_conditions=fail_conditions)
    matches=d.run()
    return d, matches


def test_allmatch(li):
    d=TestSubdict(li, li)
    matches=d.run()
    assert len(set(matches.values()))==1, matches
    return d, matches


testlist=['Afonso, Alexandria C. F.',
        'Afonso, A.',
 'Afonso, A. S.' ,
 'Afonso, AM',
 'Afonso, Ana', 
 'Afonso, Ana Isabel',
 'Afonso, Ana M.',]


testlist2=[
        'Afonso, A.',

 'Afonso, AM',
 'Afonso, Ana', 
 'Afonso, Ana M.',
 'Afonso, A. M.']


testlist3=['Mendez, V. Ernesto', 
           'Mendez, Victor Ernesto', 
           'Mendez, VE', 
           'Mendez, V. E.']

def test_main(data_path, n=10000):
    df=pd.read_csv(data_path).head(n)
    df['AF']=df['AF'].astype(str).apply(titleCase_LN)
    a_list=make_alist(df['AF'])
    dict_1=initial_matches(a_list)
    matches=solved_to_point(dict_1)
    return matches, a_list, dict_1

def main(data_path, save_path):
    print('Matching Authors')
    df=pd.read_csv(data_path)    
    df['AF']=df['AF'].astype(str).apply(titleCase_LN)
    a_list=make_alist(df['AF'])
    dict_1=initial_matches(a_list)
    matches=solved_to_point(dict_1)
    df.to_csv(save_path, index=False)
    return matches, a_list, dict_1
#%%
if __name__=='__main__':
    d, matches=test_set(testlist, {'Afonso, A.': 'Afonso, A. S.'})
    d, matches=test_allmatch(testlist2)
    d, matches=test_allmatch(testlist3)
    d, matches=test_set(['Adeyemi, Olutobi', 'Adeyemi, Omowumi O.'], 
                        )
    data_path=os.path.join('data', 'all_data.csv')
    save_path=os.path.join('data', 'corrected_authors.csv')
    matches, alist, dict1=main(data_path, save_path)

