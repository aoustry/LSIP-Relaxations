# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:17:05 2021

@author: aoust
"""

import QPInstance
import MosekRelaxationSolver 
import QPLIBReader



################################### Sequence of instructions for a given instance#######################################
def execute(name):
    parser = QPLIBReader.QPLIBReader("qplib/"+name)
    LB,UB = parser.bounds() 
    print("Files parsed.")
    n = parser.n
    Instance = QPInstance.QPInstance(name, n, parser.objective_polynomial(), parser.nonnegative_constraints_polynomials(), UB, LB,[],False)
    BO = Instance.generateBoundOracle()
    print("Oracles created.")
    f = Instance.objectiveArray()
    G = Instance.G_MC_Tr_matrix(False)
    SDPoracles = Instance.SDPOraclesOnly()
    R = MosekRelaxationSolver.MosekRelaxationSolver(len(f),f,BO.UB,BO.LB,G,SDPoracles,name)
    R.build_model_and_solve()

################################# Applying the instructions for all instances ########################################
for name in ["QPLIB_5922", "QPLIB_5935","QPLIB_5944","QPLIB_8505","QPLIB_8559", "QPLIB_8991","QPLIB_10002"]:
    try:
        execute(name+".qplib")
    except Exception as inst:
        f = open("output_mosek/"+name+"_mosek.csv","w")
        f.write(str(inst))
        f.close()