# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:57:36 2021

@author: aoust
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:04:06 2020

@author: aoust
"""
import QPInstance
import MosekRelaxationSolver
import FileReader



################################### Sequence of instructions for a given instance#######################################
def execute(name,i):
    obj = "insts_rand/{0}_obj.dat".format(name)
    bounds = "insts_rand/{0}_bounds.dat".format(name)
    FReader = FileReader.FileReader(obj, bounds)
    Instance = QPInstance.QPInstance(name, FReader.n, FReader.objective_polynomial, FReader.inequality_polynomials, FReader.UB, FReader.LB,[],False)
    BO = Instance.generateBoundOracle()
    print(type(BO.indices))
    print("Oracles created.")
    f = Instance.objectiveArray()
    G = Instance.G_MC_Tr_matrix(False)
    SDPoracles = Instance.SDPOraclesOnly()
    R = MosekRelaxationSolver.MosekRelaxationSolver(len(f),f,BO.UB,BO.LB,G,SDPoracles,name)
    R.build_model_and_solve()

################################# Applying the instructions for all instances ########################################

for i in [1,2,3,4,6,7]:
    name = "rand"+str(i)
    try:
        execute(name,i)
    except Exception as inst:
        f = open("output_mosek/"+name+"_mosek.csv","w")
        f.write(str(inst))
        f.close()
