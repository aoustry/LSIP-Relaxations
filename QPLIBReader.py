# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 09:56:21 2020

@author: aoust
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:48:36 2020

@author: aoust
"""
from QuadraticPolynomial import QuadraticPolynomial
import numpy as np
import os
import pandas as pd

class QPLIBReader():
    
    def __init__(self,filename):
        
        self.f  = open(filename)
        self.name = str(self.f.readline()).replace("\n","")
        self.instance_type = str(self.f.readline()).replace("\n","")
        self.sense = self.f.readline().replace("\n","")
        line = self.f.readline()
        self.n = int(line.split("#")[0])
        self.__objective_polynomial = QuadraticPolynomial(self.n,[],[])
        self.__constraints_polynomials = []
       
        #Parsing
        self.__read_m()
        self.__read_obj_quad_part()
        self.__read_obj_lin_part()
        self.__read_obj_const_part()
        
        self.__read_const_quad_part()
        self.__read_const_lin_part()
        
        self.__read_infinite_value()
        self.__read_const_bounds()
        
        self.__read_bounds()
        
        self.__read_var_types()
        
                
        #Processing
        self.__set_binary_bounds()
        
        #Security check
        self.__check_pol()
        self.__check_bounds()
        
        
        #Closing file
        self.f.close()
        print("Nb const polynomial = {0}".format(len(self.__constraints_polynomials)))
    
    def is_treatable(self):
        return (-self.infinity<min(self.__LB)) and (self.infinity>max(self.__UB))
        
    def objective_polynomial(self):
        assert(self.sense in ["minimize","maximize"])
        if self.sense=="minimize":
            return self.__objective_polynomial
        else:
            newPoly = QuadraticPolynomial(self.n,list(self.__objective_polynomial.tuples), np.array(list(-self.__objective_polynomial.coefs)))
            return newPoly
        
    def nonnegative_constraints_polynomials(self):
        """Return the nonnegative polynomials defining the set of constraints """
        
        output = []
        #Constraints polynomials
        for i in range(len(self.__constraints_polynomials)):
            poly = self.__constraints_polynomials[i]
            if self.__constraints_LHS[i]>-self.infinity:
                new_poly = QuadraticPolynomial(poly.n, list(poly.tuples), list(poly.coefs))
                new_poly.tuples.append((-1,-1))
                new_poly.coefs.append(-self.__constraints_LHS[i])
                new_poly.check()
                output.append(new_poly)
            if self.__constraints_RHS[i]<self.infinity:
                new_poly = QuadraticPolynomial(poly.n, list(poly.tuples), list(-poly.coefs))
                new_poly.tuples.append((-1,-1))
                new_poly.coefs.append(self.__constraints_RHS[i])
                new_poly.check()
                output.append(new_poly)
        
        #TO DELETE: Binary constraints as quadratic constraints
        # for i in range(self.n):
        #     if self.__type_var[i]==2:
        #         new_poly = QuadraticPolynomial(self.n, [(-1,i),(i,i)], [1,-1])
        #         new_poly.check()
        #         output.append(new_poly)
        #         new_poly = QuadraticPolynomial(self.n, [(-1,i),(i,i)], [-1,1])
        #         new_poly.check()
        #         output.append(new_poly)
        return output
    
    def bounds(self):
        return self.__LB, self.__UB
        
    def __read_m(self):
        self.m = 0
        if not(self.instance_type[2] in ["N","B"]):
            line = self.f.readline()
            sp = (line.split("#"))
            assert("constraints" in sp[1])
            self.m = int(sp[0])
            for i in range(self.m):
                self.__constraints_polynomials.append(QuadraticPolynomial(self.n,[],[]))
            
    def __read_obj_quad_part(self):
        if not(self.instance_type[0] in ["L"]):
            line = self.f.readline()
            sp = (line.split("#"))
            assert("quadratic terms in objective" in sp[1])
            terms_number = int(sp[0])
            for k in range(terms_number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==3)
                i,j, coef = int(split[0]),int(split[1]), float(split[2]) 
                assert(i>=j)
                assert(j>=1)
                # Offset in variables
                self.__objective_polynomial.tuples.append((j-1,i-1))
                #Reminder : the quadratic term is 1/2* x^TQx
                if (i>j):
                    self.__objective_polynomial.coefs.append(coef)
                else:
                    assert(i==j)
                    self.__objective_polynomial.coefs.append(0.5*coef)
            
    
    def __read_obj_lin_part(self):
        line = self.f.readline()
        sp = (line.split("#"))
        assert("default value for linear coefficients in objective" in sp[1])
        defaut_value = float(sp[0])
        line = self.f.readline()
        sp = (line.split("#"))
        assert("number of non-default linear coefficients in objective" in sp[1])
        terms_number = int(sp[0])
        if defaut_value!=0.0:
            b = [defaut_value] * self.n
            for k in range(terms_number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==2)
                i, coef = int(split[0]),float(split[1])
                assert(i>=1)
                b[i-1] = coef
                
            self.__objective_polynomial.coefs.extend(b)
            tup = [(-1,i) for i in range(self.n)]
            self.__objective_polynomial.tuples.extend(tup)
        else:
            for k in range(terms_number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==2)
                i, coef = int(split[0]),float(split[1])
                assert(i>=1)
                assert(i<=self.n)
                self.__objective_polynomial.coefs.append(coef)
                self.__objective_polynomial.tuples.append((-1,i-1))
    
    
    def __read_obj_const_part(self):
        line = self.f.readline()
        sp = (line.split("#"))
        assert("objective constant" in sp[1])
        obj_constant = float(sp[0])
        self.__objective_polynomial.coefs.append(obj_constant)
        self.__objective_polynomial.tuples.append((-1,-1))
        
        
    def __read_const_quad_part(self):
        
        if not(self.instance_type[2] in ["N","B","L"]):
            line = self.f.readline()
            sp = (line.split("#"))
            assert("number of quadratic terms in all constraints" in sp[1])
            terms_number = int(sp[0])
            for k in range(terms_number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==4)
                idx,i,j,coef = int(split[0]),int(split[1]),int(split[2]),float(split[3])
                assert((idx<=self.m) and (idx>=1))
                assert((i<=self.n) and (i>=1))
                assert((j<=self.n) and (j>=1))
                assert(i>=j)
                self.__constraints_polynomials[idx-1].tuples.append((j-1,i-1))
                if (i>j):
                    self.__constraints_polynomials[idx-1].coefs.append(coef)
                else:
                    self.__constraints_polynomials[idx-1].coefs.append(0.5*coef)
                    
                    
    def __read_const_lin_part(self):
        if not(self.instance_type[2] in ["N","B"]):
            line = self.f.readline()
            sp = (line.split("#"))
            assert("number of linear terms in all constraints" in sp[1])
            terms_number = int(sp[0])
            for k in range(terms_number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==3)
                idx,i,coef = int(split[0]),int(split[1]),float(split[2])
                assert((idx<=self.m) and (idx>=1))
                assert((i<=self.n) and (i>=1))
                self.__constraints_polynomials[idx-1].tuples.append((-1,i-1))
                self.__constraints_polynomials[idx-1].coefs.append(coef)
                
    def __read_infinite_value(self):
        
        line = self.f.readline()
        sp = line.split("#")
        assert("value for infinity" in sp[1])
        self.infinity = float(sp[0])
        
    def __read_const_bounds(self):
        
        if not(self.instance_type[2] in ["N","B"]):
            
            #LHS of constraints
            line = self.f.readline()
            sp = (line.split("#"))
            assert("default left-hand-side value" in sp[1])
            default_value_lb = float(sp[0])
            self.__constraints_LHS = [default_value_lb]*self.m
            line = self.f.readline()
            sp = (line.split("#"))
            assert("number of non-default left-hand-sides" in sp[1])
            number = int(sp[0])
            for k in range(number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==2)
                idx,bound = int(split[0]),float(split[1])
                assert((idx<=self.m) and (idx>=1))
                self.__constraints_LHS[idx-1] = bound
                

            #RHS of constraints
            line = self.f.readline()
            sp = (line.split("#"))
            assert("default right-hand-side value" in sp[1])
            default_value_ub = float(sp[0])
            self.__constraints_RHS = [default_value_ub]*self.m
            line = self.f.readline()
            sp = (line.split("#"))
            assert("number of non-default right-hand-sides" in sp[1])
            number = int(sp[0])
            for k in range(number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==2)
                idx,bound = int(split[0]),float(split[1])
                assert((idx<=self.m) and (idx>=1))
                self.__constraints_RHS[idx-1] = bound
            
    
    def __read_bounds(self):
        
        if not(self.instance_type[1] in ["B"]):
            #Lower-bounds
            line = self.f.readline()
            sp = (line.split("#"))
            assert("default variable lower bound value" in sp[1])
            default_value_lb = float(sp[0])
            self.__LB = [default_value_lb]*self.n
            line = self.f.readline()
            sp = (line.split("#"))
            assert("number of non-default variable lower bounds" in sp[1])
            number = int(sp[0])
            for k in range(number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==2)
                idx,bound = int(split[0]),float(split[1])
                assert((idx<=self.n) and (idx>=1))
                self.__LB[idx-1] = bound
                

            #Upper_bounds
            line = self.f.readline()
            sp = (line.split("#"))
            assert("default variable upper bound value" in sp[1])
            default_value_ub = float(sp[0])
            self.__UB = [default_value_ub]*self.n
            line = self.f.readline()
            sp = (line.split("#"))
            assert("number of non-default variable upper bounds" in sp[1])
            number = int(sp[0])
            for k in range(number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==2)
                idx,bound = int(split[0]),float(split[1])
                assert((idx<=self.n) and (idx>=1))
                self.__UB[idx-1] = bound
                
        else:
            self.__LB = [0]*self.n
            self.__UB = [1]*self.n
            
    def __read_var_types(self):
        
        if not(self.instance_type[1] in ["B","C","I"]):
            line = self.f.readline()
            sp = (line.split("#"))
            assert("default variable type" in sp[1])
            default_var_type = int(sp[0])
            self.__type_var = [default_var_type]*self.n
            line = self.f.readline()
            sp = (line.split("#"))
            assert("number of non-default variable types" in sp[1])
            number = int(sp[0])
            for k in range(number):
                line = self.f.readline()
                split = line.split(" ")
                assert(len(split)==2)
                idx, typevar = int(split[0]), int(split[1])
                assert((idx<=self.n) and (idx>=1))
                self.__type_var[idx-1] = typevar
        
        else:
            if self.instance_type[1] == "C":
                self.__type_var = [0]*self.n
            if self.instance_type[1] == "I":
                self.__type_var = [1]*self.n
            if self.instance_type[1] == "B":
                self.__type_var = [2]*self.n
        
        self.binary_variables = set()
        for i in range(self.n):
            if self.__type_var[i]==2:
                self.binary_variables.add(i)
            
    def __check_pol(self):
        self.__objective_polynomial.check()
        for poly in self.__constraints_polynomials:
            poly.check()
            assert(not((-1,-1) in poly.tuples))
            
    def __check_bounds(self):
        
        for i in range(self.n):
            assert(self.__LB[i]<=self.__UB[i])
            
            
    def __set_binary_bounds(self):
        for i in range(self.n):
            if self.__type_var[i]==2:
                self.__UB[i] = 1.0
                self.__LB[i] = 0.0
                


                

            
        
# folder =   "QPLIB/qplib/html/qplib"
# files = os.listdir(folder)            
# count = 0
# sizes = []
# names = []
# for f in files:
#     print(f)
#     P = QPLIBparser(folder+"/"+f)
#     if P.is_treatable():
#         sizes.append(P.n)
#         names.append(P.name)
# df = pd.DataFrame()
# df["name"] = names
# df["size"] = sizes
# df.to_csv("instances.csv")
       
        
        

        
        
        
        
        
    