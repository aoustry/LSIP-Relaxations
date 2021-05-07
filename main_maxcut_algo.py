# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:04:06 2020

@author: aoust
"""
import QPInstance
import ALROMPSolver 
import FiniteConstraintRelaxationSolver
import MaxCutReader
import numpy as np

###################################Parameters###########################################################################
def eps_function(i):
    return 2*(0.95**i)
cinit =  1
max_iter = 300
inner_max_iter = 10000
max_total_iterations = 10000
K = 500
folder = "output_algo"

################################### Sequence of instructions for a given instance#######################################
def execute(filepath, name):

    reader = MaxCutReader.MaxCutReader(filepath)
    f = open(name+"half_sum_weights.txt","x")
    f.write(str(reader.half_sum_weights))
    f.close()
    n = reader.n
    Instance = QPInstance.QPInstance(name, n, reader.objective_polynomial, reader.constraint_polynomials, [1]*n, [-1]*n,set(),True,set(range(n)))
    M = Instance.generateMasterOracle(False,with_bound_oracle= True, with_triangle_ineq = True)
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
for i in range(0,10):
    name = "w09_100.{0}".format(i)
    filepath = "data/rudy/"+name    
    execute(filepath,name)