#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:48:18 2021

@author: bdube
"""

import unittest
from author_work import *
import pandas as pd

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

class TestsMatchers:
    def test_not_match1(self):
        d, matches=test_set(testlist, {'Afonso, A.': 'Afonso, A. S.'})
    
    def test_notmatch2(self):
        d, matches=test_set(['Adeyemi, Olutobi', 'Adeyemi, Omowumi O.'], 
                               {'Adeyemi, Olutobi': 'Adeyemi, Omowumi O.',
                               'Adeyemi, Omowumi O.': 'Adeyemi, Olutobi',})
    def test_allmatch1(self):
        d, matches=test_allmatch(testlist2)
    def test_allmatch2(self):
        d, matches=test_allmatch(testlist3)
    
       