# LSIP-Relaxations
Linear Semi-Infinite Programming Relaxations for Polynomial Programming problems

## Dependencies 



The required Python 3 packages for the AL-ROMP solver implementation are:

- numpy
- scipy
- chompack
- cvxopt
- pandas
- osqp

For the benchmarks, the packages docplex (CPLEX API) and Fusion (MOSEK API) are also required.


## Executing the code


The files main_xxx_algo.py give exemples of command to execute the AL-ROMP solver on the different types of instances. The files main_xxx_mosek.py give exemples of command to execute the MOSEK solver on the different types of instances, in the case of SDP instances.

## Scaling of the instances

The objective function of the instances is scaled, and the results of the AL-ROMP algorithm and of the MOSEK solver are presented in the "new" scaling. The scaling factor is available in the files describing the caracteristics of the instances (in the folder "instances_characteristics/").

---------------------------------------------------------------------------------------
## Affiliations

(o) LIX CNRS, École Polytechnique, Institut Polytechnique de Paris, 91128, Palaiseau, France 

(o) École des Ponts, 77455 Marne-La-Vallée

---------------------------------------------------------------------------------------

Sponsored by Réseau de transport d’électricité, 92073 La Défense, France
