# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:04:14 2020

@author: aoust
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:27:22 2020

@author: aoust
"""
import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from Oracle import MT_ListOracle,ListOracle,BoundOracle,partialSDPOracle, McCormickOracle,FiniteOracle, TriangleOracle
from scipy.sparse import vstack
from cvxopt import spmatrix, amd
import chompack as cp
import random
from itertools import combinations
from QuadraticPolynomial import QuadraticPolynomial

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

class QPInstance():
    
    def __init__(self, name, n, objective_polynomial, inequality_polynomials, upper_bounds, lower_bounds,binary_variables,fulldense=False,tr_ineq_variables = []):
        
        self.name = name
        self.n = n
        self.fulldense = fulldense
        self.objective_polynomial = objective_polynomial
        self.inequality_polynomials = inequality_polynomials
        self.upper_bounds = np.array(upper_bounds)
        self.lower_bounds = np.array(lower_bounds)
        self.binary_variables = binary_variables
        self.tr_ineq_variables = tr_ineq_variables
        #Add polynomials defining binary variables
        for i in self.binary_variables:
            new_poly = QuadraticPolynomial(self.n, [(-1,i),(i,i)], [1,-1])
            new_poly.check()
            self.inequality_polynomials.append(new_poly)
            new_poly = QuadraticPolynomial(self.n, [(-1,i),(i,i)], [-1,1])
            new_poly.check()
            self.inequality_polynomials.append(new_poly)
            
        self.__compute_edges_and_cliques(fulldense)
        
        if len(self.edges)>0:
            for cl in self.cliques:
                assert(strictly_increasing(cl))
            self.__cliques_completion()
            self.__rescale()
            self.log_instance_characteristics()
            self.compute_var_to_clique()
            self.reversed_edges = {value : key for (key, value) in self.edges.items()}
            self.BOmarkers = [self.clique_containing_monomials([1+i]) for i in range(2*self.n+len(self.edges))]
            self.MCmarkers =  [self.clique_containing_var([i,j]) for (i,j) in self.edges]
        self.N = 1+2*self.n+len(self.edges)
        
    def compute_var_to_clique(self):
        self.var_to_clique = [[]]*self.n
        for k in range(len(self.cliques)):
            cl = self.cliques[k]
            for i in cl:
                assert(i<self.n)
                self.var_to_clique[i].append(k)
                
    def clique_containing_var(self,variables):
        variables = list(variables)
        candidates= set(self.var_to_clique[variables[0]])
        for i in variables:
            assert(i<self.n)
            candidates = candidates.intersection(self.var_to_clique[i])
        assert(len(candidates)>0)
        return (random.sample(candidates,1))[0]
    
    def clique_containing_monomials(self,monomials):
        var = set()
        for m in monomials:
            if m>=1 and m<=self.n:
                var.add(m-1)
            if (m>=self.n+1) and (m<=2*self.n):
                var.add(m-self.n-1)
            if m>=2*self.n+1:
                edge = m-(2*self.n+1)
                i,j = self.reversed_edges[edge]
                var.add(i)
                var.add(j)
        return self.clique_containing_var(var)
                
    def __compute_edges_and_cliques(self,fulldense = False):
        
        if fulldense == False:
            I,J = [i for i in range(self.n)],[i for i in range(self.n)]
            V = [1.0 for i in range(self.n)]
        self.edges = {}
        count = 0
        for a,b in self.objective_polynomial.vpairs():
            assert(a<b)
            if not((a,b) in self.edges):
                self.edges[(a,b)] = count
                count+=1
                if fulldense == False:
                    I.append(a)
                    J.append(b)
                    I.append(b)
                    J.append(a)
                    V.append(1.0)
                    V.append(1.0)
       
        for poly in self.inequality_polynomials:
            
            for a,b in poly.vpairs():
                assert(a<b)
                if not((a,b) in self.edges):
                    self.edges[(a,b)] = count
                    count+=1
                    if fulldense == False:
                        I.append(a)
                        J.append(b)
                        I.append(b)
                        J.append(a)
                        V.append(1.0)
                        V.append(1.0)
        
        
        if len(self.edges)>0:
            if fulldense == False:
                csp_graph = spmatrix(V, I, J, (self.n,self.n))
                
                symb = cp.symbolic(csp_graph, p=amd.order)
                (symb.sparsity_pattern(reordered=False))
                
                self.cliques = symb.cliques(reordered=False)
                self.clean_cliques()
            else:
                self.cliques = [[i for i in range(self.n)]]
        

    def __find_monomial(self,i,j):
        assert(i<=j)
        if i==-1:
            if j==-1:
                return 0
            else:
                return 1+j
        else:
            if i==j:
                return 1+self.n+i
            else:
                return 1+2*self.n+self.edges[(i,j)]
            
    def __cliques_completion(self):
        count = len(self.edges)
        for clique in self.cliques:
            for i,j in combinations(clique, 2):
                if not((i,j) in self.edges):
                    assert(i<j)
                    self.edges[(i,j)] = count
                    count+=1
        self.N = 1 + 2*self.n + len(self.edges)
        
    def __rescale(self):
        
        self.M = np.array([max(abs(self.upper_bounds[i]),abs(self.lower_bounds[i])) for i in range(self.n)])
        self.M = np.where(self.M==0, 0.0001, self.M)
        self.upper_bounds = np.true_divide(self.upper_bounds,  self.M)
        self.lower_bounds = np.true_divide(self.lower_bounds,  self.M)
        self.objective_polynomial.scale_variables(self.M)
        self.obj_factor = self.objective_polynomial.scale_coefs2()
        
        for poly in self.inequality_polynomials:
            poly.scale_variables(self.M)
            poly.scale_coefs()
            
            
    def __compute_linear_constraint_matrix(self):
        assert(len(self.inequality_polynomials)>0)
        temp_poly = []
        markers = []
        for poly in self.inequality_polynomials:
            ind_list = []
            coef_list = []
            for i,j,coef in poly.enumerate_triples():
                monom = self.__find_monomial(i, j)
                ind_list.append(monom)
                coef_list.append(coef)
            zeros = [0 for i in range(len(ind_list))]
            coefs = np.array(coef_list)
            marker = self.clique_containing_monomials(ind_list)
            temp_poly.append(coo_matrix((coefs,(zeros,ind_list)),shape = (1,self.N)).tocsr())
            markers.append(marker)
        return vstack(temp_poly), markers
    
    def clean_cliques(self):
        new_cliques = []
        for cl in self.cliques:
            if len(cl)>=2:
                cl.sort()
                new_cliques.append(cl)
        self.cliques = new_cliques
            
    def objectiveArray(self):
        ind_list = []
        coef_list = []
        for i,j,coef in self.objective_polynomial.enumerate_triples():
                
                monom = self.__find_monomial(i, j)
                ind_list.append(monom)
                coef_list.append(coef)
                
        zeros = [0 for i in range(len(ind_list))]
        coefs = np.array(coef_list)
        res = coo_matrix((coefs,(zeros,ind_list)),shape = (1,self.N)).toarray()
        return np.reshape(res, (self.N,))
    
    
    def generateBoundOracle(self):
        L = len(self.edges)
        indices = [1+i for i in range(2*self.n+L)]
        lbo = np.array(list(self.lower_bounds)+[0 for i in range(self.n)]+[-1 for i in range(L)])
        ubo = np.array(list(self.upper_bounds)+[max(self.lower_bounds[i]**2,self.upper_bounds[i]**2) for i in range(self.n)]+[1 for i in range(L)])
        BO = BoundOracle(self.N,indices,lbo,ubo, self.BOmarkers)
        return BO
    
    def generateTriangleIneqTriplets(self):
        neighbours = {}
        for i in self.tr_ineq_variables:
            neighbours[i] = []
        for a,b in self.edges:
            assert(a<b)
            if a in self.tr_ineq_variables and b in self.tr_ineq_variables:
                assert(a in neighbours)
                neighbours[a].append(b)
        edge_triplets = []
        ind_offset = 1 + 2*self.n
        for a in self.tr_ineq_variables:
            for b in neighbours[a]:
                for c in neighbours[b]:
                    if (a,c) in self.edges:
                        assert(a<b)
                        assert(b<c)
                        edge_triplets.append((ind_offset+self.edges[(a,b)],ind_offset+self.edges[(b,c)],ind_offset+self.edges[(a,c)]))
        return edge_triplets 
                
    
    def generateMasterOracle(self, MT, with_bound_oracle = False,with_triangle_ineq = False):
        
        #Bound oracle
        L = len(self.edges)
        indices = [1+i for i in range(2*self.n+L)]
        assert(self.lower_bounds.min()>=-1)
        assert(self.upper_bounds.max()<=1)
        
        if MT:
            masterOracle = MT_ListOracle(self.N,[])
        else:
            masterOracle = ListOracle(self.N,[])
        
        if with_bound_oracle:
            lbo = np.array(list(self.lower_bounds)+[0 for i in range(self.n)]+[-1 for i in range(L)])
            ubo = np.array(list(self.upper_bounds)+[max(self.lower_bounds[i]**2,self.upper_bounds[i]**2) for i in range(self.n)]+[1 for i in range(L)])
            BO = BoundOracle(self.N,indices,lbo,ubo, self.BOmarkers)
            masterOracle.addOracle(BO)
        
        #MCormick Oracle
        AUX = np.array([[1+i,1+j,1+2*self.n+self.edges[(i,j)]] for (i,j) in self.edges])
        AUX = AUX.T
        
        if len(AUX)>0:
            MCO = McCormickOracle(self.N,np.array(self.lower_bounds),np.array(self.upper_bounds),AUX[0].astype(int),AUX[1].astype(int),AUX[2].astype(int),self.MCmarkers)
            masterOracle.addOracle(MCO)
            
        if with_triangle_ineq:
            triplets = self.generateTriangleIneqTriplets()
            L = len(triplets)
            print("Len triplets triangle = "+str(L))
            masterOracle.addOracle(TriangleOracle(self.N, triplets))
            
        
        if len(self.inequality_polynomials)>0:
            G, clique_markers = self.__compute_linear_constraint_matrix()
            self.Gmarkers = clique_markers
            G_Oracle = FiniteOracle(self.N,len(self.inequality_polynomials),G, clique_markers)
            masterOracle.addOracle(G_Oracle)
       
        for idx_clique in range(len(self.cliques)):
            clique = self.cliques[idx_clique]
            marker = idx_clique
            matrixSize = len(clique)+1
            # if matrixSize == self.n+1:
            #     assert(self.fulldense)
            vector_indices = [0] + [1+i for i in clique] + [1+self.n+i for i in clique] + [1+2*self.n+self.edges[(i,j)] for (i,j) in combinations(clique, 2)]
            y_submatrix_indices = [0] + [1+idx for idx in range(len(clique))] + [1+idx for idx  in range(len(clique))] + [1+clique.index(j) for (i,j) in combinations(clique, 2)]
            x_submatrix_indices = [0] +     [0 for idx in range(len(clique))] + [1+idx for idx  in range(len(clique))] + [1+clique.index(i) for (i,j) in combinations(clique, 2)]
            masterOracle.addOracle(partialSDPOracle(self.N, matrixSize,vector_indices, x_submatrix_indices, y_submatrix_indices, marker))
        return masterOracle 
    
    def generateSDPOracleForErrorBound(self):
        clique = list(range(self.n))
        matrixSize = len(clique)+1
        marker = 0 #arbitrary
        vector_indices = [0] + [1+i for i in clique] + [1+self.n+i for i in clique] + [1+2*self.n+self.edges[(i,j)] for (i,j) in self.edges]
        y_submatrix_indices = [0] + [1+idx for idx in range(len(clique))] + [1+idx for idx  in range(len(clique))] + [1+clique.index(j) for (i,j) in self.edges]
        x_submatrix_indices = [0] +     [0 for idx in range(len(clique))] + [1+idx for idx  in range(len(clique))] + [1+clique.index(i) for (i,j) in self.edges]
         
        return partialSDPOracle(self.N, matrixSize,vector_indices, x_submatrix_indices, y_submatrix_indices,marker)
    
    def log_instance_characteristics(self):
        aux = len(self.inequality_polynomials)
        f = pd.DataFrame()
        f["n"] = [self.n]
        f["N"] = [self.N]
        f["Number of ineq. constraints"] = [aux]
        f["Objective scaling factor"] = [self.obj_factor]
        f["MinLb"] = [self.lower_bounds.min()]
        f["MaxUb"] = [self.upper_bounds.max()]
        f["cliques number "] = [len(self.cliques)]
        f.to_csv("instances_characteristics/"+self.name+"instance_params.csv",index=False)
        

    def G_and_MCmatrix(self):
        result = []
        markers = []
        if len(self.inequality_polynomials)>0:
            markers = list(self.Gmarkers)
            for poly in self.inequality_polynomials:
                ind_list = []
                coef_list = []
                for i,j,coef in poly.enumerate_triples():
                    monom = self.__find_monomial(i, j)
                    ind_list.append(monom)
                    coef_list.append(coef)
                zeros = [0 for i in range(len(ind_list))]
                coefs = np.array(coef_list)
                result.append(coo_matrix((coefs,(zeros,ind_list)),shape = (1,self.N)).tocsr())
                
        for i,j in self.edges:
            assert(i<j)
            assert(j<self.n)
            li,lj,ui,uj = self.lower_bounds[i],self.lower_bounds[j],self.upper_bounds[i],self.upper_bounds[j]
            k = self.edges[(i,j)]
            x = [0, 1+i, 1+j, 1+2*self.n+k]
            y = np.zeros(4)
            for q in range(4):
                if q==0:
                    coef = [li*lj,-lj,-li,1.0]
                if q==1:
                    coef = [-li*uj,uj,li,-1.0]
                if q==2:
                    coef = [-lj*ui,lj,ui,-1.0]
                if q ==3:
                    coef = [ui*uj,-uj,-ui,1.0]
                sg = coo_matrix((coef, (y, x)), shape=(1,self.N))
                result.append(sg)
                markers.append(self.MCmarkers[k])
        return vstack(result), markers
    
    
    def G_MC_Tr_matrix(self,with_triangle_ineq):
        result = []
        if len(self.inequality_polynomials)>0:
            for poly in self.inequality_polynomials:
                ind_list = []
                coef_list = []
                for i,j,coef in poly.enumerate_triples():
                    monom = self.__find_monomial(i, j)
                    ind_list.append(monom)
                    coef_list.append(coef)
                zeros = [0 for i in range(len(ind_list))]
                coefs = np.array(coef_list)
                result.append(coo_matrix((coefs,(zeros,ind_list)),shape = (1,self.N)).tocsr())
                
        for i,j in self.edges:
            assert(i<j)
            assert(j<self.n)
            li,lj,ui,uj = self.lower_bounds[i],self.lower_bounds[j],self.upper_bounds[i],self.upper_bounds[j]
            k = self.edges[(i,j)]
            x = [0, 1+i, 1+j, 1+2*self.n+k]
            y = np.zeros(4)
            for q in range(4):
                if q==0:
                    coef = [li*lj,-lj,-li,1.0]
                if q==1:
                    coef = [-li*uj,uj,li,-1.0]
                if q==2:
                    coef = [-lj*ui,lj,ui,-1.0]
                if q ==3:
                    coef = [ui*uj,-uj,-ui,1.0]
                sg = coo_matrix((coef, (y, x)), shape=(1,self.N))
                result.append(sg)
                
                
        if with_triangle_ineq:
            edge_triplet_list = self.generateTriangleIneqTriplets()
            A = [trip[0] for trip in edge_triplet_list]
            B = [trip[1] for trip in edge_triplet_list]
            C = [trip[2] for trip in edge_triplet_list]
            for k in range(len(A)):
                x = [0, A[k], B[k], C[k]]
                
                for q in range(4):
                    if q==0:
                        coef = [1.0,1.0,1.0,1.0]
                    if q==1:
                        coef = [1.0,1.0,-1.0,-1.0]
                    if q==2:
                        coef = [1.0,-1.0,1.0,-1.0]
                    if q ==3:
                        coef = [1.0,-1.0,-1.0,1.0]
                    sg = coo_matrix((coef, (y, x)), shape=(1,self.N))
                    result.append(sg)
    
        return vstack(result)
    
    
    def SDPOraclesOnly(self):
        liste = []
        for idx_clique in range(len(self.cliques)):
            clique = self.cliques[idx_clique]
            marker = idx_clique
            matrixSize = len(clique)+1
            # if matrixSize == self.n+1:
            #     assert(self.fulldense)
            vector_indices = [0] + [1+i for i in clique] + [1+self.n+i for i in clique] + [1+2*self.n+self.edges[(i,j)] for (i,j) in combinations(clique, 2)]
            y_submatrix_indices = [0] + [1+idx for idx in range(len(clique))] + [1+idx for idx  in range(len(clique))] + [1+clique.index(j) for (i,j) in combinations(clique, 2)]
            x_submatrix_indices = [0] +     [0 for idx in range(len(clique))] + [1+idx for idx  in range(len(clique))] + [1+clique.index(i) for (i,j) in combinations(clique, 2)]
            liste.append(partialSDPOracle(self.N, matrixSize,vector_indices, x_submatrix_indices, y_submatrix_indices, marker))
        return liste