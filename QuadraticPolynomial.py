# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:37:07 2020

@author: aoust
"""

import numpy as np
import math

class QuadraticPolynomial():
    
    def __init__(self,n,tuples, coefs):
        
        self.n = n
        assert(len(tuples)==len(coefs))
        self.tuples = tuples
        self.coefs = coefs
        for (i,j) in tuples:
            assert(i<=j)
            
    def check(self):
        for (i,j) in self.tuples:
            
            assert(i<=j)
        
        if type(self.coefs)==list:
            self.coefs = np.array(self.coefs) 
        
    def vpairs(self):
        
        for (i,j) in self.tuples:
            if ((i>=0) and (i<j)):
                yield i,j

    def scale_variables(self,tab):
        for k in range(len(self.tuples)):
            i,j = self.tuples[k]
            factor = 1
            if i!=-1:
                factor = factor*tab[i]
            if j!=-1:
                factor = factor*tab[j]
            self.coefs[k] = self.coefs[k]*factor
    
    def scale_coefs(self):
        self.coefs = self.coefs/(np.linalg.norm(self.coefs,2))
        
    
    def scale_coefs2(self):
        power = int(math.log10(np.linalg.norm(self.coefs,2)))
        factor = 10**(power-1)
        self.coefs = self.coefs/factor
        return factor  
        
    def enumerate_triples(self):
        for k in range(len(self.tuples)):
            i,j = self.tuples[k]
            c = self.coefs[k]
            yield i,j,c
    
    def variables_list(self):
        set_of_variables = set()
        for (i,j) in self.tuples:
            if i!=-1:
                set_of_variables.add(i)
            if j!=-1:
                set_of_variables.add(j)
        res = list(set_of_variables)
        res.sort()
        return res
    
    def evaluation(self,x):
        S = 0
        for k in range(len(self.tuples)):
            i,j = self.tuples[k]
            S+=x[i]*x[j]*self.coefs[k]
        return S