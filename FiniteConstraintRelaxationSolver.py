# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:25:20 2020

@author: aoust
"""


from docplex.mp.advmodel import AdvModel
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

class FiniteConstraintRelaxationSolver():
    
    
    def __init__(self,N,f,UB,LB,G,markersUB,markersLB, markersG):
        assert(len(f)==N)
        assert(len(UB)==N-1)
        assert(len(LB)==N-1)
        assert(len(markersUB)==N-1)
        assert(len(markersLB)==N-1)
        assert(len(markersG)==G.shape[0])
        self.N,self.f,self.UB,self.LB,self.G = N,f,[1] + list(UB),[1] + list(LB),G
        self.markersUB,self.markersLB, self.markersG = [-1] + list(markersUB), [-1] + list(markersLB), markersG
        assert(len(self.UB)==N)
        assert(len(self.LB)==N)
        
    def computeRelaxation(self):
        mdl = AdvModel('my model')
        y = mdl.continuous_var_list(self.N,lb = self.LB,ub = self.UB)
        
        obj_expr = mdl.sum(self.f[i] * y[i] for i in range(self.N))
        mdl.minimize(obj_expr)
        y0_constraint = (y[0] == 1)
        mdl.add_constraint(y0_constraint)
        ub_constraints = (y[i] <= self.UB[i] for i in range(self.N))
        mdl.add_constraints(ub_constraints)
        
        lb_constraints = (y[i] >= self.LB[i] for i in range(self.N))
        mdl.add_constraints(lb_constraints)
        
        Gconstraints = mdl.matrix_constraints(self.G, y, np.zeros(self.G.shape[0]), 'GE')
        mdl.add_constraints(Gconstraints)
        
        mdl.solve()
        return (mdl.objective_value)
    
    def computeRelaxationAndsolutions(self):
        mdl = AdvModel('my model')
        y = mdl.continuous_var_list(self.N,lb = self.LB,ub = self.UB)
        
        obj_expr = mdl.sum(self.f[i] * y[i] for i in range(self.N))
        mdl.minimize(obj_expr)
        y0_constraint = (y[0] == 1)
        mdl.add_constraint(y0_constraint)
        ub_constraints = [y[i] <= self.UB[i] for i in range(self.N)]
        mdl.add_constraints(ub_constraints)
        
        lb_constraints = [y[i] >= self.LB[i] for i in range(self.N)]
        mdl.add_constraints(lb_constraints)
        
        Gconstraints = mdl.matrix_constraints(self.G, y, np.zeros(self.G.shape[0]), 'GE')
        mdl.add_constraints(Gconstraints)
        
        mdl.solve()
        
        y_opt = [y[i].solution_value for i in range(self.N)]
        dual_ub = mdl.dual_values(ub_constraints)
        dual_lb = mdl.dual_values(lb_constraints)
        dual_G = mdl.dual_values(Gconstraints)
        support = []
        weights = []
        markers = []
        
        for i in range(1,self.N):
            assert(dual_ub[i]>=-0.0001)
            if abs(dual_ub[i]) > 10E-7:
                y = np.zeros(2)
                x = np.array([0, i])
                coefs = np.array([self.UB[i],-1])
                vect = (coo_matrix(((coefs), (y, x)), shape=(1, self.N)))
                support.append(vect.to_csr())
                weights.append(dual_ub[i])
                markers.append(self.markersUB[i])
                
        for i in range(1,self.N):
            assert(dual_lb[i]<=0.0001)
            if abs(dual_lb[i]) > 10E-7:
                y = np.zeros(2)
                x = np.array([0, i])
                coefs = np.array([-self.LB[i],1])
                vect = (coo_matrix(((coefs), (y, x)), shape=(1, self.N)))
                support.append(vect.to_csr())
                weights.append(-dual_lb[i])
                markers.append(self.markersLB[i])
                
        for i in range(self.G.shape[0]):
            assert(dual_G[i]>=-0.0001)
            if abs(dual_G[i])>=10E-7:
                support.append(csr_matrix(self.G.getrow(i)))
                weights.append(dual_G[i])
                markers.append(self.markersG[i])
            
        
        return (mdl.objective_value), y_opt, support, np.array(weights), markers
    
    
    def computeRelaxationAndsolutions_noboundsinsol(self):
        mdl = AdvModel('my model')
        y = mdl.continuous_var_list(self.N,lb = self.LB,ub = self.UB)
        
        obj_expr = mdl.sum(self.f[i] * y[i] for i in range(self.N))
        mdl.minimize(obj_expr)
        y0_constraint = (y[0] == 1)
        mdl.add_constraint(y0_constraint)
        ub_constraints = [y[i] <= self.UB[i] for i in range(self.N)]
        mdl.add_constraints(ub_constraints)
        
        lb_constraints = [y[i] >= self.LB[i] for i in range(self.N)]
        mdl.add_constraints(lb_constraints)
        
        Gconstraints = mdl.matrix_constraints(self.G, y, np.zeros(self.G.shape[0]), 'GE')
        mdl.add_constraints(Gconstraints)
        
        mdl.solve()
        
        y_opt = [y[i].solution_value for i in range(self.N)]
        dual_ub = mdl.dual_values(ub_constraints)
        dual_lb = mdl.dual_values(lb_constraints)
        dual_G = mdl.dual_values(Gconstraints)
        support = []
        weights = []
        markers = []
        
                      
        for i in range(self.G.shape[0]):
            assert(dual_G[i]>=-0.0001)
            if abs(dual_G[i])>=10E-7:
                support.append(csr_matrix(self.G.getrow(i)))
                weights.append(dual_G[i])
                markers.append(self.markersG[i])
            
        
        return (mdl.objective_value), y_opt, support, np.array(weights), markers
    
    def computeRelaxationAndLog(self,filename):
        
        self.val = self.computeRelaxation()
        f = pd.DataFrame()
        f["Linear Relaxation Value"] = [self.val]
        f.to_csv(filename+"LR.csv",index=False)