# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:04:06 2020

@author: aoust
"""
import QPInstance
import MosekRelaxationSolver
import SparseReader
import numpy as np


################################### Sequence of instructions for a given instance#######################################
def execute(filepath, name):

    reader = SparseReader.SparseReader(filepath)
    n = reader.n
    Instance = QPInstance.QPInstance(name, n, reader.objective_polynomial, [], [1]*n, [0]*n,set(range(n)),True)
    BO = Instance.generateBoundOracle()
    print("Oracles created.")
    f = Instance.objectiveArray()
    G = Instance.G_MC_Tr_matrix(False)
    SDPoracles = Instance.SDPOraclesOnly()
    R = MosekRelaxationSolver.MosekRelaxationSolver(len(f),f,BO.UB,BO.LB,G,SDPoracles,name)
    R.build_model_and_solve()


################################# Applying the instructions for all instances ########################################
for i in range(1,6):
    name = "gka{0}f.sparse".format(i)
    filepath = "biqmaclib/gka/"+name   
    try:
        execute(filepath,name)
    except Exception as inst:
        f = open("output_mosek/"+name+"_mosek.csv","w")
        f.write(str(inst))
        f.close()
