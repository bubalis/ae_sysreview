# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:52:06 2020

@author: benja
"""

import pandas as pd
import re
import numpy as np
import csv
import json

#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process

def aut_cleaner(string):
    '''CLean author names'''
    string=re.sub(r'"|\[|', '', string)
    return re.sub(r"'|\]", '', string).strip()
    
def list_auts(paper_auts):
    '''Return all unique author names in string.'''
    autFind=re.compile(r'\[.+?\]') #author names are in brackets.
    authors=autFind.findall(str(paper_auts))
    return [aut_cleaner(author).split(';') for author in authors]    

def list_flattener(nested_list):
    '''Flatten a nested list of any depth to a flat list'''
    while True:
        nested_list=[n for x in nested_list for n in x]
        if type(nested_list[0])!=list:
            return nested_list
        
def list_all_authors(series):
    '''Return a list of all authors, given a series representing the authors.'''    
    aut_nested_list=series.tolist()
    aut_nested_list= [a for a in aut_nested_list if type(a)==str]
    flat_list= list_flattener([section.split(';') for section in aut_nested_list])
    return list(set([item.strip() for item in flat_list]))


def LN_to_titlecase(df, aut_col):
    as_list=df[aut_col].to_list()
    df[aut_col]=[LN_to_titlecaseReplacer(multi_name_string) for multi_name_string in as_list]
    return df

def LN_to_titlecaseReplacer(multi_name_string):
    '''Replace all names in a multi_name string with same name but author as title_case.'''
    if type(multi_name_string)!=str:
        return multi_name_string
    string_as_list=str(multi_name_string).split(';')
    for item in string_as_list:
        multi_name_string=multi_name_string.replace(item.split(',')[0], item.split(',')[0].title())
    return multi_name_string
    
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
    '''Return a dictionary of lists. 
     {LastName First Letter: [All names with that LastName FirstLetter]}'''
    f_letter =re.compile(r'^.+?\,\s\w')
    dic={} 
    a_list.sort()
    match_list=[]
    i=0
    while i<len(a_list):
        author=a_list[i]
        if not match_list:
            if f_letter.search(author):
                LN_firstletter=f_letter.search(author).group()
                match_list.append(author)
                i+=1
            else:
                i+=1
        else:
            if LN_firstletter in author:
                match_list.append(author)
                i+=1
            else:
                dic[LN_firstletter]=match_list
                match_list=[]
    dic[LN_firstletter]=match_list
    return dic

        
#%%
def initial_matches(name_list):
    '''Get first round of matches for names.'''
    init_dict=initial_dic(name_list)
    return {k:LNgroup(v, name_list).find_matches() for k,v in init_dict.items() if len(v)>1}
    


name_spliter_re=re.compile(r'\,|\s')
def name_spliter(name):
    '''Split name into its component parts.'''    
    return [part.strip() for part in name_spliter_re.split(name) if part]
    

           

def name_part_type(string):
    '''Categorize a name Component'''
    if re.search('[A-Z]\.', string) and string[-1]=='.':
        return name_part_codes[2]
    elif len(string)>1 and re.search('[A-Z]{2,3}', string):
        return name_part_codes[3]
    elif len(string)==1:
        return name_part_codes[4]
    elif re.search('[A-Z]\-[A-Z]', string):
        return name_part_codes[5]
        
    elif string:
        return name_part_codes[1]
    
    else:
        return name_part_codes[6]


    
class WordPartCode(object):
    '''Meta-class for word part codes'''
    pass

class FullName(WordPartCode):
    pass

class InitialWDot(WordPartCode):
    pass

class List_of_Initials(WordPartCode):
    
    def fix(nameObj):
        '''Turn initials of form Doe, JE to: Doe, J. E.'''
            
        match=re.search('[A-Z]{2,4}', nameObj.out_name).group()
        replacement='. '.join([char for char in match])+'.'
        return nameObj.string.replace(match, replacement) 

class InitialNoDot(WordPartCode):
    def fix(nameObj):
        '''Turn initials of form Doe, J E to:
            Doe, J. E.'''
        matches=re.findall(r'[A-Z](?:\s+|$)', nameObj.out_name)
        new_string=nameObj.out_name
        for match in matches:
            new_string=re.sub(f'{match}(?=\\b)', f'{match[0]}.', new_string)
            
        return new_string

class InitialWDash(WordPartCode):
    def fix(nameObj):
        '''Turn initials of form Doe, J-E to:
            Doe, J. E.'''
        match=re.search('[A-Z\-]{2,5}', nameObj.out_name).group()
        replacement='. '.join([char for char in match if char!='-'])+'.'
        return nameObj.out_name.replace(match, replacement)



name_part_codes={ # number codes for 
                 1: FullName(),
                 2: InitialWDot(),
                 3: List_of_Initials(),
                 4: InitialNoDot(),
                 5: InitialWDash(),
                 6: None
        }


match_criteria={'full_string': 'exclusive', 
                'fn_match':   'longest',
                'all_initials': 'exclusive',
                 'two_initials_default': 'exclusive',
                 'dash_replace': 'exclusive',
                 'one_initial_default': 'exclusive'}



def instance_in_list(obj, li):
    return any([isinstance(x, obj) for x in li])

def check_all_instance(items, classes):
    if len(items)==len(classes):
        return all([isinstance(items[i], Class) for i, Class in enumerate(classes)])
    return False

class Name(object):
    '''Object storing a author's Name. 
    String: the original form of the name.
    ln_group: the LNgroup object that holds all name objects with matching LastName First Initials
    out_name: What form of the name will this name be changed to {default is self.string}'''
    
    def __init__(self, string, ln_group=[]):
        self.string=string
        string=self.clean()
        self.update(string)
        self.resolve_misformat_initials()
        self.ln_group=ln_group
    
    def clean(self):
        if "'" in self.string or '"' in self.string:
            return self.string.replace("'", '').replace('"', '')
        return self.string
    
    def assign_to_match(self, matches, match_criterion):
        '''Take matches and update name object based on these matches.'''
        if not matches:
            return
        elif match_criterion=='exclusive':
            if len(matches)==1:
                new_name=matches.pop()
            else:
                return
        elif match_criterion=='longest':
            new_name=max(matches, key=len)
        if len(self.out_name)<len(new_name):
            self.update(new_name)
                
    def function_from_string(self, string):
        '''Run matching method based on its string name.'''
        method = getattr(self, string)
        matches=method()
        match_criterion=match_criteria[string]
        self.assign_to_match(matches, match_criterion)
        
    def update(self, string):
        self.parts=name_spliter(string)
        self.part_codes=tuple([name_part_type(part) for part in self.parts])
        self.out_name=string
        
       
    def resolve_misformat_initials(self):
        '''Reformat Initials for Consistency'''
        for partClass in [List_of_Initials, InitialNoDot, InitialWDash]:
            if instance_in_list(partClass, self.part_codes):
                self.update(partClass.fix(self))
                break
    
    def retrieve_parent(self):
        '''If name obj's out_name is pointing to a name that itself is pointing to a different name, upedate to that name.'''
        if self.string!=self.out_name:
            match=next((x for x in self.ln_group.names if x.string == self.out_name), None)
            if match:
                return match.retrieve_parent()
        return self.out_name
   
    def full_string(self):
        '''Return any exact matches for the full string of name.'''
        if not instance_in_list(InitialWDot, self.part_codes):
            return {name.out_name for name in self.ln_group.names if (self.out_name in  name.out_name and name!=self)}
         
    def all_initials(self):
        '''Return any matches where all name parts match either initial to full, full to initial or full to full.'''
        matches={name.out_name for name in self.ln_group.names if (name_parts_matcher(self, name) and name!=self)}
        if len({name.out_name for name in self.ln_group.names if self.part_codes==name.part_codes})==1:
            return matches
    
    def fn_match(self):
        '''Return all Matches where Lastname and Firstname Match 
        and all other elements are at least an initials match.'''
        return {name.out_name for name in self.ln_group.names if
                 all([name!=self,
                      self.parts[1]==name.parts[1],
                      name_parts_matcher(self, name, start_index=2)
                         ])}
     
    def two_initials_default(self):
        '''If name is of form (Last Name, Initial, Initial) 
        and there is only one possible match that isn't ruled out, return that match. '''
        if check_all_instance(self.part_codes,(FullName, InitialWDot, InitialWDot)):
            if all ([check_all_instance(name.part_codes, (FullName, FullName)) 
            for name in self.ln_group.names if name.out_name!=self.out_name]):
                matches={name.out_name for name in self.ln_group.names if name.out_name!=self.out_name}
                return matches
        
    
    def one_initial_default(self):
        if check_all_instance(self.part_codes, (FullName, InitialWDot)):
            if all ([check_all_instance(name.part_codes, (FullName, FullName)) 
            for name in self.ln_group.names if name.out_name!=self.out_name]):
                matches={name.out_name for name in self.ln_group.names if name.out_name!=self.out_name}
                return matches
        
        
    def dash_replace(self):
        '''Return all matches created by replacing - with space or nothing.'''
        if '-' in self.string:
            substitutions=[self.string.replace('-', ' '), self.string.replace('-', '')]
            return {name.out_name for name in self.ln_group.names 
                    if any([substitution.lower()==name.out_name.lower()
                        for substitution in substitutions])}

#%%
def name_parts_matcher(name, potential_match, start_index=1):
    '''Return whether all parts of two names are
    The same full string 
    or if 1 inital,
    The same initial'''
    if len(name.parts)<=len(potential_match.parts) and len(name.parts)>start_index:
        for i in range(start_index, len(name.parts)):
            i_part_codes=[name.part_codes[i], potential_match.part_codes[i]]
            if any([isinstance(x,InitialWDot) for x in i_part_codes]):
                match_result=initialMatch(name.parts[i], potential_match.parts[i])
                if not match_result:
                    return False
            elif all([isinstance(x, FullName) for x in i_part_codes]):
                match_result= name.parts[i]==potential_match.parts[i]
                if not match_result:
                    return False
        return True
    return False
            
def initialMatch(part1, part2):
    '''Return whether first letter of two strings is the same.'''
    return part1[0]==part2[0]

class LNgroup(object):
    '''Take a list of names that share  Lastname, First initial and match them using various methods.
    Can exlude some functions to make matching less aggressive'''
    

    def __init__(self, sub_list, full_list, exclude=[]):
        self.full_list=full_list
        self.names=tuple([Name(author, self) for author in sub_list])
        self.flags=exclude
        
    def __repr__(self):
        return f'LN group object with the following current matches: \n {self.make_dict()}'
    
    
    def make_dict(self):
        '''Return a dictionary of {old_name: out_name for all names in group}'''
        dictionary={name.string: name.out_name for name in self.names}
        assert all([name.string in dictionary.keys() for name in self.names])
        return dictionary
    
    def match_all(self, function):
        '''Match all names in the LN_group using the function name passed as string.'''
        for name in self.names:
            name.function_from_string(function)
    
    def update_all(self):
        '''Update all names to point to the 'terminal' out_name.'''
        for name in self.names:
            name.retrieve_parent()
    
    def find_matches(self):
        '''Match using all available methods.'''
        for method in ['dash_replace',
                       'full_string', 
                       'fn_match',
                       'one_initial_default',
                       'all_initials',
                       'two_initials_default'
                       ]:
            self.match_all(method)
            self.update_all()
        return self.make_dict()
        
    
def parts_matcher_tester(str1, str2, start_index):
    
    name1=Name(str1)
    name2=Name(str2)
    return name_parts_matcher(name1, name2, start_index)
    
def solved_to_point(nested_dict):
    dictionary={}
    for v in {k:v for k,v in nested_dict.items()}.values():
        dictionary={**dictionary, **{key: value for key,value in v.items() if key!=value}}
    return dictionary

def not_solved_yet(nested_dict, solved_dict, name_list):
    '''Return a dictionary of all sub_dicts that are still ambiguous, 
    have at least one name in one or two initial form, and have at least one value in the solve dictionary.'''
    return {k:v for k,v in nested_dict.items() 
            if (len(set(v.values()))>1) 
            and (any(p in solved_dict.values() for p in v.values()))
            and (re.search(r'[^\s]+\,\s\w\.(\s\w\.|$)', '$'.join(v.values())))} 
    
    
def make_matches(df, aut_col, author_inst_col, other_cols=[]):
    df=LN_to_titlecase(df, aut_col)
    author_list=list_all_authors(df[aut_col])
    dict_1=initial_matches(author_list)
    matches_1=solved_to_point(dict_1)
    df[aut_col]=[dict_find_replace(au, matches_1, split_char=';') for au in df[aut_col].to_list()]
    return matches_1, df, author_list

test_list=list({ 'Aerts, JM': 'Aerts, J. M.',
 'Aerts, R.': 'Aerts, R. J.',
 'Aerts, RJ': 'Aerts, R. J.',
 'Aerts, Raf': 'Aerts, R. J.',
 'Aerts, Rien': 'Aerts, R. J.',}.keys())



class TestLNgroup(LNgroup):
    

def test(test_list):
    group=LNgroup(test_list, test_list)
    print(group.find_matches())

test(['Aerts, JM', 'Aerts, R.', 'Aerts, RJ', 'Aerts, Raf', 'Aerts, Rien'])

test(['Adeyemi, Olutobi', 'Adeyemi, Omowumi O.'])

topics={'ecological_intensification':['ecological intensification', ''], 
        'sustainable_intensification': ["sustainable intensification", ' '],
        'climate_smart': ['climate-smart|climate smart| climatesmart', 'agriculture'],
        'sustainable_agriculture': ['sustainable agriculture',' '],
       'agroecology': ['agroecology|agro-ecology', ' '],
       'ecoagriculture':['ecoagriculture|eco-agriculture'],
        'alternative_agriculture':['alternative agriculture'],
        'permaculture': ['permaculture|perma-culture'],
        'ipm':['integrated pest management', ' '],
        'biodynamics': ['biodynamic|bio-dynamic', 'agriculture'],
        'regenerative agriculture': ['regenerative agriculture'],
        'conservation agriculture': ['conservation agriculture'],
        'organic agriculture': ['organic agriculture'],
        
                         }

def topic_col_maker(v):
    kw_abstract_cols=['TI', 'DE', 'ID', 'AB', 'MA', 'SC']
    def specific_func(row):
        regexes=[re.compile(v_part.lower()) for v_part in v]
        return int(
        any(
        [ 
         all([reg.search( str(row[col]).lower()) for reg in regexes ])
            for col in kw_abstract_cols]
        ))
                
        
       
    return specific_func
    
    



#%%  
def add_TopicColumns(df, topics):
    for k, v in topics.items():
        function=topic_col_maker(v)
        df[k]=df.apply(function, axis=1)
    return df

def main(df):
    print('Matching Authors')
    matches, df, a_list =make_matches(df, 'AF', 'C1')
    return df, a_list, matches, {k: v for k, v in globals().items()}

def save_results(matches, df, a_list):
    save_path=os.path.join('Data',  'corrected_authors.csv'
    df.to_csv(save_path)
    with open(os.path.join('Data','author_matches.csv'), 'w') as f:
        print(json.dumps(matches), file=f)
    new_alist=list_all_authors(df['AF'])
    with open(os.path.join('Data','author_list.txt'), 'w') as f:
        print('\n'.join(a_list), file=f)

if __name__=='__main__':
    #df,  a_list, matches, other_results=main(pd.read_csv(r"C:\Users\benja\sysreviewfall2019\Data\all_data.csv"))
    #save_results(matches, df, a_list)
        
    
    