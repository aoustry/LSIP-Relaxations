# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:30:32 2020

@author: aoust
"""

from Bundle import Bundle
import numpy as np
import qpsolvers
import time
import pandas as pd
import FiniteConstraintRelaxationSolver
from docplex.mp.advmodel import AdvModel
from Oracle import ConstantOracle
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import eye
import osqp, math
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import vstack, diags

BigM = 1.0E9
dense = False
epsilon = 0
print("Dense Gram matrix computation = {0}".format(dense))
print("WARNING EPSILON = {0}".format(epsilon))


class ALROMP_RelaxationSolver():
    
    def __init__(self,n,N, f, S_Oracle,name, Bound_oracle,innerK,max_total_iterations,folder):
        self.n = n
        self.name = name
        self.N = N
        self.f = f
        assert(f[0]==0)
        self.S_Oracle = S_Oracle
        self.boundOracleAndSupport = Bound_oracle
        self.initialized = False
        self.innerK = innerK
        self.max_total_iterations = max_total_iterations
        self.folder = folder
        
        self.outer_iteration_log = []
        self.inner_iteration_log= []
        self.OracleTime_log= []
        self.QP_time_log= []
        self.LB1_log= []
        self.bestLB1_log= []
        self.LB2_log= []
        self.bestLB2_log= []
        self.valQP_log = []
        self.L1norm_log = []
        self.L2norm_log = []
        self.clog= []
        self.score_log = []
        self.bound_time_log = []
        
        
    
    def log(self,boundtime,oracle_time, qptime,score):
        self.outer_iteration_log.append(self.outer_iteration)
        self.inner_iteration_log.append(self.inner_iteration)
        self.OracleTime_log.append(oracle_time)
        self.bound_time_log.append(boundtime)
        self.QP_time_log.append(qptime)
        self.LB1_log.append(self.LB1)
        self.bestLB1_log.append(self.bestLB1)
        self.LB2_log.append(self.LB2)
        self.bestLB2_log.append(self.bestLB2)
        self.valQP_log.append(self.valQP)
        self.L1norm_log.append(self.L1norm)
        self.L2norm_log.append(self.L2norm)
        self.clog.append(self.__c)
        self.score_log.append(score)
        
    
    def savelog(self):
        output = pd.DataFrame()
        output["outer_it"] = self.outer_iteration_log 
        output["inner_it"] =self.inner_iteration_log
        output["OracleTime"] =self.OracleTime_log
        output["QPtime"] =self.QP_time_log
        output["Bound time"] =self.bound_time_log
        output["bestLB1"] =self.bestLB1_log
        output["LB1"] =self.LB1_log
        output["bestLB2"] =self.bestLB2_log
        output["LB2"] =self.LB2_log
        output["valQP"] = self.valQP_log
        output["c"] =self.clog
        output["score"] = self.score_log
        output["L1norm"] = self.L1norm_log
        output["L2norm"] = self.L2norm_log
        
        output.to_csv(self.folder+"/log_ALCG_"+self.name+".csv")
        
    
    def initialize(self, y, vectors, weights, markers):
        assert(y[0]==1)
        assert(len(vectors)==len(weights))
        self.initialized = True
        self.y = np.array(y)
        support = Bundle(self.N,maintainGramMatrix  = False,first_abs = 1)
        support.add(vectors, markers)
        support.updateWeights(weights)
        self.qs = support.aggregation()
        self.qb = np.zeros(self.N)
        self.q = self.qs + self.qb
    
                
    def classicalAugmentedLagrangian(self,cinit,tol_function,max_iter,inner_max_iter):
        self.full_counter = 0
        
        self.__c = cinit
        self.mu = 1/self.__c
        self.bestLB1 = -10E9
        self.LB2 = self.bestLB2 = -10E9
        if self.initialized == False:
            self.qs = np.zeros(self.N)
            self.qb = np.zeros(self.N)
            self.q = self.qs + self.qb
            self.y = np.zeros(self.N)
            self.y[0] = 1
                           
        for i in range(max_iter):
            assert(self.y[0]==1)
            self.outer_iteration = i
            tol = tol_function(i)
            starting = 50
            if i-starting>1:
                self.__c = self.__c * 1.05
                self.mu = 1/self.__c
            
            print("Outer iteration nb {0}".format(i))
            print("Tolerance {0}".format(tol))
            self.ROMP(tol,inner_max_iter)
            grad = self.compute_gradient()
            self.y = grad
            
            if (self.full_counter>=self.max_total_iterations):
                self.savelog()
                return
        self.savelog()
            
    def compute_gradient(self):
        delta = self.q - self.f
        delta[0] = 0
        return self.y + self.__c * delta
    
    def ROMP(self, tol, inner_max_iter):
        
        
        for i in range(inner_max_iter):
            print("----------Iteration #{0}#{1}------------------------".format(self.outer_iteration,i))
            print("c = {0}".format(self.__c))
            self.inner_iteration = i
                       
            #Main descent
            gradient = self.compute_gradient()
            if i>0:
                dist = np.linalg.norm(gradient-self.y)
                self.LB2 = gradient.dot(self.f) + self.mu * (dist**2) - 2*self.mu*math.sqrt(len(self.f))*dist
                self.bestLB2 = max(self.bestLB2,self.LB2)
            t0 = time.time()
            scores = self.S_Oracle.computeScores(gradient, self.innerK)
            cost = abs(scores[0])
            external_vectors, markers = self.S_Oracle.retrieve(self.innerK)
            oracle_time = time.time() - t0
            self.full_counter+=1
            running = ((cost>tol) or (i==0) )and(self.full_counter < self.max_total_iterations)
            if running:
                t0 = time.time()
                self.multidimensional_exact_descent(external_vectors, 'OSQP')
                qptime = time.time() - t0
            else:
                qptime = 0

            print("Ybar value = {0}".format(self.y.dot(self.f)))
            print("Tolerance  = {0} / Cost = {1}".format(tol,cost))
            
            self.log(0,oracle_time, qptime,cost)
            
            if i%20==5:
                self.savelog()
            
            if not(running):
                return False
            
            print("LB1 = {0}".format(self.LB1))
            print("LB2 = {0}".format(self.LB2))
            
            
        return True
            

    def multidimensional_exact_descent(self,vectors, solver):
        
        if solver == "OSQP":
            coo_matrix_qs = coo_matrix((self.qs, (np.zeros(self.N),np.arange(self.N))), shape=(1,self.N))
            csr_matrix_qs = coo_matrix_qs.tocsr()
            coo_matrix_qb = coo_matrix((self.qb, (np.zeros(self.N),np.arange(self.N))), shape=(1,self.N))
            csr_matrix_qb = coo_matrix_qb.tocsr()
            P = vstack([csr_matrix_qs,csr_matrix_qb]+ vectors)
            
            #Objective: Linear part
            aux = self.y - self.__c*self.f
            aux[0] = self.y[0]            
            linear_part = P.dot(aux)
            
            #Objective: quadratic part
            mask = np.ones(self.N)
            mask[0] =0
            mask_mat = diags(mask)
            Q = (P.dot(mask_mat)).dot(P.transpose())
            
            #Constraints
            var_nb = 2+len(vectors)
            A = scipy.sparse.eye(var_nb,var_nb)
            l = np.zeros(var_nb)
            u = np.array([np.inf for i in range(var_nb)])
            solver = osqp.OSQP()
        
            solver.setup(P=self.__c*Q, q=linear_part, A=A, l=l, u=u,max_iter = 4000,eps_rel = 1E-9, eps_prim_inf=1E-9, eps_dual_inf=1E-9,warm_start =True,verbose=False,polish=True)
            solver.warm_start(x = np.array([1,1]+[0]*len(vectors)))
            results = solver.solve()
            
            alpha = results.x
            #qtest = (P.transpose()).dot(alpha)
            assert(alpha[1]>-10E-5)
            
            
            #Aggregating the rest
            alpha[1] = 0
            self.qs = (P.transpose()).dot(alpha)
            self.q = self.qb+self.qs
            #assert(np.linalg.norm(qtest-self.q)<0.0001)
            delta = self.f - self.q
            delta[0] = 0
            valdual = 0.5*self.__c*np.linalg.norm(delta,2)**2 + (-delta).dot(self.y) + (self.q[0] - self.f[0])
            self.valQP = -valdual
            self.LB1 = self.f[0]-self.q[0]-np.linalg.norm(delta,1)
            self.bestLB1 = max(self.LB1, self.bestLB1)
            self.L1norm = np.linalg.norm(delta,1)
            self.L2norm = np.linalg.norm(delta,2)
            fcopy = np.copy(self.f)
            fcopy[0] = 0
            offset = -(self.f).dot(self.y) + 0.5*self.__c*np.linalg.norm(fcopy,2)**2
            print("Val dual = {0}".format(-valdual))
                        
            if (abs(results.info.obj_val  + offset - valdual)>0.0001):
                print("ALERT : GAP = {0}".format(abs(results.info.obj_val  + offset - valdual)))
            
            
            