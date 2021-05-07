# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:04:06 2020

@author: aoust
"""
import QPInstance
import ALROMPSolver
import FiniteConstraintRelaxationSolver
import FileReader
import numpy as np

###################################Parameters###########################################################################
def eps_function(i):
    return 2*(0.95**i)
cinit = 0.05
max_iter = 300
inner_max_iter = 10000
max_total_iterations = 10000
K = 500
folder = "output_algo"

################################### Sequence of instructions for a given instance#######################################
def execute(name,i):
    obj = "insts_rand/{0}_obj.dat".format(name)
    bounds = "insts_rand/{0}_bounds.dat".format(name)
    FReader = FileReader.FileReader(obj, bounds)
    Instance = QPInstance.QPInstance(name, FReader.n, FReader.objective_polynomial, FReader.inequality_polynomials, FReader.UB, FReader.LB,[],False)
    
    M = Instance.generateMasterOracle(Instance.N>50000,with_bound_oracle= True, with_triangle_ineq = False)
    BO = Instance.generateBoundOracle()
    print(type(BO.indices))
    print("Oracles created.")
    f = Instance.objectiveArray()
    G, MMarkers = Instance.G_and_MCmatrix()
    fs= FiniteConstraintRelaxationSolver.FiniteConstraintRelaxationSolver(Instance.N,f,BO.UB,BO.LB,G, BO.markers, BO.markers, MMarkers)
    fs.computeRelaxationAndLog(name+"_lp.txt")
    print("Linear relaxation solved.")
    solver = ALROMPSolver.ALROMP_RelaxationSolver(Instance.n,len(f),f,M,name,BO,K,max_total_iterations,folder)
    solver.classicalAugmentedLagrangian(cinit,eps_function,max_iter,inner_max_iter)
   

################################# Applying the instructions for all instances ########################################

for i in [1,2,3,4,5,6,7]:
    name = "rand"+str(i)
    execute(name,i)
