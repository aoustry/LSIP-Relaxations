# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:48:36 2020

@author: aoust
"""
from QuadraticPolynomial import QuadraticPolynomial
import numpy as np

class FileReader():
    
    
    def __init__(self,fileobj, filebounds, fileeq =None, fileineq=None):
        
        self.__read_obj(fileobj)
        self.__read_bounds(filebounds)
        self.inequality_polynomials = []
        if fileeq!=None:
            self.__read_poly(fileeq,True)
        if fileineq!=None:
            self.__read_poly(fileineq,False)
    
    def __read_obj(self, fileobj):
        file = open(fileobj)
        #Reading n
        self.n = int(file.readline())
        #Reading e
        e = int(file.readline())
        #Reading l
        tup = []
        coefs =  []
        for i in range(self.n):
            tup.append((-1,i))
            coefs.append(float(file.readline()))
        #Reading q
        for i in range(self.n):
            assert(i==int(file.readline()))
            assert(i==int(file.readline()))
            tup.append((i,i))
            coefs.append(float(file.readline()))
        #Reading b
       
        for i in range(e):
            k = int(file.readline())
            l = int(file.readline()) 
            assert(k<l)
            tup.append((k,l))
            coefs.append(float(file.readline()))
        file.close()
        self.objective_polynomial = QuadraticPolynomial(self.n,tup,np.array(coefs))
            
   
    def __read_bounds(self,fbounds):
        file = open(fbounds)
        self.UB, self.LB = [],[]
        assert(self.n == int(file.readline()))
        for i in range(self.n):
            self.LB.append(float(file.readline()))
        for i in range(self.n):
            self.UB.append(float(file.readline()))
            assert(self.LB[i]+0.01<=self.UB[i])
        file.close()
    
    def __read_poly(self,filepoly, eq):
        
        file = open(filepoly)
        nb_const = int(file.readline())
        for i in range(nb_const):
            s = int(file.readline())
            tup_list = []
            coef_list = []
            for k in range(s):
                line = file.readline()
                split = line.split(" ")
                i,j, coef = int(split[0]),int(split[1]), float(split[2])               
                tup_list.append((i,j))
                coef_list.append(coef)
            self.inequality_polynomials.append(QuadraticPolynomial(self.n,tup_list,np.array(coef_list)))
            if eq:
                self.inequality_polynomials.append(QuadraticPolynomial(self.n,tup_list,-np.array(coef_list)))
                