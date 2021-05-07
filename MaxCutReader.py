# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:27:32 2021

@author: aoust
"""
import numpy as np
from QuadraticPolynomial import QuadraticPolynomial

class MaxCutReader():
    
    def __init__(self,fileobj):
        
        file = open(fileobj)
        #Reading n and m
        line = (file.readline())
        array = line.split(" ")
        self.n,self.m = int(array[0]), int(array[1])
        btup = []
        bcoefs =  []
        half_sum_weights = 0
        for k in range(self.m):
            line = (file.readline())
            array = line.split(" ")
            i,j, val = int(array[0]), int(array[1]), float(array[2])
            print(i,j,val)
            btup.append((i-1,j-1))
            assert(i<j)
            bcoefs.append(0.5*val)
            half_sum_weights+=0.5*val
            
        coefs = bcoefs
        tuples = btup
        self.half_sum_weights = half_sum_weights
        self.objective_polynomial = QuadraticPolynomial(self.n,tuples,np.array(coefs))
        self.constraint_polynomials = [QuadraticPolynomial(self.n,[(-1,-1),(i,i)],[1,-1]) for i in range(self.n)] + [QuadraticPolynomial(self.n,[(-1,-1),(i,i)],[-1,1]) for i in range(self.n)] 