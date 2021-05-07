# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:39:41 2021

@author: aoust
"""

import sys
import random
import mosek
import math
from  mosek.fusion import *
import numpy as np
from scipy import linalg
from scipy.sparse import coo_matrix
import time


class MosekRelaxationSolver():
    
    def __init__(self,N,f,UB,LB,G,SDPoracles,name):
        
        self.N,self.f,self.UB,self.LB,self.G = N,f,[1] + list(UB),[1] + list(LB),coo_matrix(G)
        self.SDPoracles = SDPoracles
        self.name = name
        
    
    def build_model_and_solve(self,offset=0):
        t0 = time.time()
        self.M  = Model("relaxation SDP")
        self.y = self.M.variable("y", Domain.inRange(self.LB,self.UB))
        self.M.objective("obj", ObjectiveSense.Minimize, Expr.dot(self.f,self.y))
        Gmatrix = Matrix.sparse(self.G.shape[0],self.G.shape[1],self.G.row, self.G.col, self.G.data)
        self.M.constraint(Expr.mul(Gmatrix,self.y),Domain.greaterThan(0.0))
        self.sdp_var = []
        for oracle in self.SDPoracles:
            self.build_sdp_cl_cons(oracle)
        self.M.setLogHandler(sys.stdout)            # Add logging
        self.cons_time = time.time()-t0
        self.M.solve()
        sol_time =  self.M.getSolverDoubleInfo("optimizerTime")
        f = open("output_mosek/"+self.name+"_mosek.csv","w")
        f.write("Primal value;Dual value;cons_time;sol time \n")
        f.write(str(offset+self.M.primalObjValue())+";"+str(offset+self.M.dualObjValue())+";"+str(self.cons_time)+";"+str(sol_time)+"\n")
        f.close()

    
    
    def build_sdp_cl_cons(self,oracle):
        
        self.sdp_var.append(self.M.variable(Domain.inPSDCone(oracle.matrixSize)))
        for k in range(len(oracle.vector_indices)):
            
            main_idx, a_idx, b_idx = oracle.vector_indices[k], oracle.x_positionsInSDP[k],oracle.y_positionsInSDP[k]
            self.M.constraint(Expr.add(Expr.neg(self.y.index(main_idx)),self.sdp_var[-1].index(a_idx,b_idx)),Domain.equalsTo(0))
        
        
        # self.M.constraint(self.sdp_var[-1].index(0,0),Domain.equalsTo(1))
        # for idx in range(size):
        #     i = cl[idx]
        #     self.M.constraint(Expr.add(Expr.neg(self.x.index(i)),self.sdp_var[-1].index(0,idx+1)),Domain.equalsTo(0))
        #     self.M.constraint(Expr.add(Expr.neg(self.x.index(i)),self.sdp_var[-1].index(idx+1,0)),Domain.equalsTo(0))
        #     self.M.constraint(Expr.add(Expr.neg(self.z.index(i)),self.sdp_var[-1].index(idx+1,idx+1)),Domain.equalsTo(0))
        #     counter+=3
        #     for idx2 in range(idx+1,size):
        #         j = cl[idx2]
        #         k = self.edges_addresses[(i,j)]
        #         assert(k!=-1)
        #         self.M.constraint(Expr.add(Expr.neg(self.y.index(k)),self.sdp_var[-1].index(idx+1,idx2+1)),Domain.equalsTo(0))
        #         self.M.constraint(Expr.add(Expr.neg(self.y.index(k)),self.sdp_var[-1].index(idx2+1,idx+1)),Domain.equalsTo(0))
        #         counter+=2
        # assert(counter==(size+1)**2)

