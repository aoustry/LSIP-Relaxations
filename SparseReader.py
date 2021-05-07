# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 11:27:32 2021

@author: aoust
"""
import numpy as np
from QuadraticPolynomial import QuadraticPolynomial

class SparseReader():
    
    def __init__(self,fileobj):
        
        file = open(fileobj)
        #Reading n and m
        line = (file.readline())
        array = line.split(" ")
        self.n,self.m = int(array[0]), int(array[1])
        tup = []
        coefs =  []
        for k in range(self.m):
            line = (file.readline())
            array = line.split(" ")
            i,j, val = int(array[0]), int(array[1]), float(array[2])
            print(i,j,val)
            tup.append((i-1,j-1))
            if i==j:
                coefs.append(val)
            else:
                assert(i<j)
                coefs.append(2*val)
        
        self.objective_polynomial = QuadraticPolynomial(self.n,tup,np.array(coefs))
        
        # m = 0
        # for idx in range(1000):
        #     x = np.random.randint(2, size=self.n)
        #     m = min(m,(self.objective_polynomial.evaluation(x)))
        # print("minimum = {0}".format(m))