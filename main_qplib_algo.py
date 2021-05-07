# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:17:05 2021

@author: aoust
"""

import QPInstance
import ALROMPSolver 
import FiniteConstraintRelaxationSolver
import QPLIBReader
import numpy as np

###################################Parameters###########################################################################
def eps_function(i):
    return 2*(0.95**i)
cinit =  1
max_iter = 3000
inner_max_iter = 10000
max_total_iterations = 10000
K = 500
folder = "output_algo"

################################### Sequence of instructions for a given instance#######################################
def execute(name):
    parser = QPLIBReader.QPLIBReader("qplib/"+name)
    LB,UB = parser.bounds() 
    print("Files parsed.")
    n = parser.n
    Instance = QPInstance.QPInstance(name, n, parser.objective_polynomial(), parser.nonnegative_constraints_polynomials(), UB, LB,[],False)
    M = Instance.generateMasterOracle(Instance.N>50000,with_bound_oracle= True, with_triangle_ineq = False)
    BO = Instance.generateBoundOracle()
    print("Oracles created.")
    f = Instance.objectiveArray()
    G, MMarkers = Instance.G_and_MCmatrix()
    fs= FiniteConstraintRelaxationSolver.FiniteConstraintRelaxationSolver(Instance.N,f,BO.UB,BO.LB,G, BO.markers, BO.markers, MMarkers)
    fs.computeRelaxationAndLog(name+"_lp.txt")
    print("Linear relaxation solved.")
    solver = ALROMPSolver.ALROMP_RelaxationSolver(Instance.n,len(f),f,M,name,BO,K,max_total_iterations,folder)
    solver.classicalAugmentedLagrangian(cinit,eps_function,max_iter,inner_max_iter)

################################# Applying the instructions for all instances ########################################
for name in ["QPLIB_5922", "QPLIB_5935","QPLIB_5944","QPLIB_8505","QPLIB_8559", "QPLIB_8991","QPLIB_10002"]:
    
    execute(name+".qplib")