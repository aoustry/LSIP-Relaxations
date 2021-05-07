# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:55:20 2021

@author: aoust
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:04:06 2020

@author: aoust
"""
import QPInstance
import MosekRelaxationSolver 
import MaxCutReader



################################### Sequence of instructions for a given instance#######################################
def execute(filepath, name):

    reader = MaxCutReader.MaxCutReader(filepath)
    n = reader.n
    Instance = QPInstance.QPInstance(name, n, reader.objective_polynomial, reader.constraint_polynomials, [1]*n, [-1]*n,set(),True,set(range(n)))
    BO = Instance.generateBoundOracle()
    f = Instance.objectiveArray()
    G = Instance.G_MC_Tr_matrix(True)
    SDPoracles = Instance.SDPOraclesOnly()
    print("Oracles created.")
    R = MosekRelaxationSolver.MosekRelaxationSolver(len(f),f,BO.UB,BO.LB,G,SDPoracles,name)
    R.build_model_and_solve(-reader.half_sum_weights/Instance.obj_factor)
    
   
################################# Applying the instructions for all instances ########################################
for i in range(0,10):
    name = "w09_100.{0}".format(i)
    filepath = "biqmaclib/rudy/"+name    
    execute(filepath,name)
    
