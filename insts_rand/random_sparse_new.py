# coding: utf-8
from networkx import *
from cvxopt import spmatrix, amd
import chompack as cp
import random

def random_sp_matrix(block_number,block_size,kwatt,p_inner,p_link):
    n = block_number*block_size
    I,J = [i for i in range(n)],[i for i in range(n)]
    V = [1.0 for i in range(n)]
    for k in range(block_number):
        offset = k*block_size
        #E = fast_gnp_random_graph(block_size,p_inner).edges()
        E = connected_watts_strogatz_graph(block_size,kwatt,p_inner).edges()
        for t in E:
            i,j = t
            I.append(offset+i)
            J.append(offset+j)
            I.append(offset+j)
            J.append(offset+i)
            V.append(1.0)
            V.append(1.0)
        for i in range(block_size):
            if random.random()<p_link:
                j = random.randint(0,block_size)
                I.append(offset+i)
                J.append((offset+block_size+j)%n)
                I.append((offset+block_size+j)%n)
                J.append(offset+i)
                V.append(1.0)
                V.append(1.0)
    return spmatrix(V, I, J, (n,n)),I,J

def list_cliques(A):
    #Liste de cliques de l'extension chordale
    symb = cp.symbolic(A, p=amd.order)
    print(symb)
    print(symb.sparsity_pattern(reordered=False))
    return symb.cliques(reordered=False) #reorder = false Super important !!!!


def generate_instance_files(block_number,block_size,kwatt,p_inner,p_link,instance_name):
    A,I,J = random_sp_matrix(block_number,block_size,kwatt,p_inner,p_link)
    print(A)
    #file 1 : problème brut
    f1 = open(instance_name+"_obj.dat","w")
    n = block_size*block_number
    assert((len(I)-n)%2==0)
    e = int((len(I)-n)/2)
    f1.write(str(n)+"\n"+str(e)+"\n")
    #Linear terms
    for i in range(n):
        f1.write(str(random.random()*2-1)+"\n")
    #Quadratic terms and bilinear terms
    for k in range(len(I)):
        i,j = I[k],J[k]
        if i<=j:
            print(i,j)
            f1.write(str(i)+"\n")
            f1.write(str(j)+"\n")
            f1.write(str(random.random()*2-1)+"\n")
    #A rajouter : CONTRAINTE DU TYPE u = xy
    f1.close()

    #file 2 : cliques de l'extension chordale avec heuristique AMD
    f2 = open(instance_name+"_cliques.dat","w")
    cliques = list_cliques(A)
    C = len(cliques)
    f2.write(str(C)+"\n")
    for cl in cliques:
        cl.sort()
        print(cl)
        nc = len(cl)
        f2.write(str(nc)+"\n")
        for i in cl:
            f2.write(str(i)+"\n")
    f2.close()

    #file3 : ineq
    f3 = open(instance_name+"_ineq.dat","w")
    f3.write("0 \n")
    # nineq = 2*n
    # f3.write(str(nineq)+"\n")
    # for i in range(n):
    #     f3.write(str(2)+"\n")
    #     f3.write(str(-1)+" "+str(-1)+" "+str(1)+"\n")
    #     f3.write(str(-1)+" "+str(i)+" "+str(1)+"\n")
    # for i in range(n):
    #     f3.write(str(2)+"\n")
    #     f3.write(str(-1)+" "+str(-1)+" "+str(1)+"\n")
    #     f3.write(str(-1)+" "+str(i)+" "+str(-1)+"\n")

    f3.close()

    f4 = open(instance_name+"_eq.dat","w")
    f4.write("0 \n")
    # neq = n
    # f4.write(str(neq)+"\n")
    # for i in range(n):
    #     f4.write(str(2)+"\n")
    #     f4.write(str(-1)+" "+str(-1)+" "+str(-1)+"\n")
    #     f4.write(str(i)+" "+str(i)+" "+str(1)+"\n")
    f4.close()

    #Bounds file
    f5 = open(instance_name+"_bounds.dat","w")
    f5.write(str(n)+"\n")
    for i in range(n):
        f5.write(str(-1)+"\n")
    for i in range(n):
        f5.write(str(1)+"\n")

    f5.close()

#generate_instance_files(10,6,5,0.3,0.8,"rand0")
#generate_instance_files(20,10,5,0.3,0.8,"rand1")
#generate_instance_files(100,10,5,0.3,0.8,"rand2")
#generate_instance_files(500,10,5,0.3,0.8,"rand3")
#generate_instance_files(500,20,5,0.3,0.8,"rand4")

#generate_instance_files(2000,10,9,1,0.1,"rand5")
#generate_instance_files(1000,50,49,0.3,0.8,"rand6")
#generate_instance_files(4000,15,9,1,0.2,"rand7") #old10
generate_instance_files(20,300,190,0.8,0.3,"rand8")
