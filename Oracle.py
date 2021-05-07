# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:51:58 2020

@author: aoust
"""
import time, heapq
from operator import itemgetter
from collections import Counter
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from multiprocessing import Pool
from random import shuffle

eigtol = 1E-5
MAX_EIGEN = 2

class Oracle():

    
    def __init__(self,N):
        self.N = N
        self.loaded = False
        self.k = 0

class BoundOracle(Oracle):
    
    def __init__(self,N,indices,LB,UB,markers):
        Oracle.__init__(self,N)
        assert(not(0 in list(indices)))
        self.indices = indices
        self.LB = LB
        self.UB = UB
        print("LBmin = {0}".format(self.LB.min()))
        print("UBmax = {0}".format(self.UB.max()))
        self.M = len(indices)
        self.markers = markers
        assert(len(indices)==len(markers))
        assert(len(indices)==len(LB))
        assert(len(indices)==len(UB))
        self.LBweights = [0] * len(LB)
        self.UBweights = [0] * len(LB)
        self.arrayUB = np.array(self.UB)
        self.arrayLB = np.array(self.LB)
        
        
    def computeScores(self,vector,k):
        self.k = min(self.M,k)
        self.loaded = True
        v = vector[self.indices]
        scores = np.concatenate([v-self.LB*vector[0],self.UB*vector[0] - v])
        self.tuples = [(i, scores[i]) for i in range(2*self.M)]
        self.tuples.sort(key=itemgetter(1))
        self.temp = vector
        return [self.tuples[j][1] for j in range(k)]
    
    def retrieve(self,k2,only_neg=False):
        assert(self.loaded)
        k2 = min(k2,self.M)
        self.loaded = False
        assert(k2<=self.k)
        self.k = 0
        result = [] 
        res_markers = []
        for j in range(k2):
            if self.tuples[j][1]<0 or only_neg==False:
                aux = self.tuples[j][0]
                if aux<self.M:
                    coefs = np.array([-self.LB[aux], 1])
                    x = np.array([0, self.indices[aux]])
                    res_markers.append(self.markers[aux])
                else:
                    coefs = np.array([self.UB[aux-self.M], -1])
                    x = np.array([0, self.indices[aux-self.M]])
                    res_markers.append(self.markers[aux-self.M])
                y = np.zeros(2)
                sg = (coo_matrix(((coefs), (y, x)), shape=(1, self.N)))
                assert(abs(sg.dot(self.temp)-self.tuples[j][1])<0.000001)
                result.append(sg.tocsr())
            
        self.tuples = []
        return result, res_markers  

    def computeScores_and_retrieve(self,vector,k):
        scores = self.computeScores(vector,k)
        ps, markers = self.retrieve(len(scores))
        return scores, ps, markers
        
    def aggregation(self):
        arrayUBweights, arrayLBweights =np.array(self.UBweights),  np.array(self.LBweights)
        offset = self.arrayUB.dot(arrayUBweights) -self.arrayLB.dot(arrayLBweights)
        rest = (arrayLBweights - arrayUBweights)
        data = np.concatenate([np.array([offset]), rest])
        x = np.concatenate([np.array([0]), self.indices])
        y = np.zeros(len(x))
        res = coo_matrix((data,(x,y)), shape =(self.N,1)).toarray().reshape(self.N)
        return res
        
    # def multiplicativeWeightUpdate(self,alpha):
    #     assert(alpha>=0)
    #     self.LBweights = [alpha* w for w in self.LBweights]
    #     self.UBweights = [alpha* w for w in self.UBweights]
        
    
    # def descent(self,gradient,c):
    #     #Setting weights for a dual move
    #     assert(gradient[0]==1)
    #     v = gradient[self.indices]
    #     LBscores = v - self.LB*gradient[0]
    #     UBscores = v - self.UB*gradient[0]
    #     for i in range(len(self.indices)):
    #         assert(abs(self.UBweights[i]*self.LBweights[i])<=1E-5)
    #         if LBscores[i]<0:
    #             assert(UBscores[i]<0)
    #             target = LBscores[i]
    #             if self.UBweights[i]>0: #On commence par diminuer
    #                 diff = min(c*self.UBweights[i], -target)
    #                 assert(diff>0)
    #                 target = target + diff
    #                 self.UBweights[i] = self.UBweights[i] - diff/c
    #                 assert(self.UBweights[i]>=-1E-5)
    #                 assert(target<=0)
    #             if target <0:
    #                 self.LBweights[i] = self.LBweights[i] -target/c
    #         if UBscores[i]>0:
    #             assert(LBscores[i]>0)
    #             target = UBscores[i]
    #             if self.LBweights[i]>0: #On commence par diminuer
    #                 diff = min(target, c*self.LBweights[i])
    #                 assert(diff>0)
    #                 target = target - diff
    #                 self.LBweights[i] = self.LBweights[i]-diff/c
    #                 assert(self.LBweights[i]>=-1E-5)
    #                 assert(target>=0)
    #             if target>0:
    #                 self.UBweights[i] = self.UBweights[i] + target/c
                            
                    
    def delta_min(self, delta):
        assert(delta[0]==0)
        assert(len(delta)==1+len(self.indices))
        delta2 = delta[self.indices]
        product1 = delta2*self.LB
        product2 = delta2*self.UB
        res = np.minimum(product1, product2)
        return res.sum()
                
        
class McCormickOracle(Oracle):
    
    def __init__(self,N,LB,UB,I_indices,J_indices,Product_Indices,markers):
        Oracle.__init__(self,N)
        assert(not(0 in list(I_indices)))
        self.LB = LB
        self.UB = UB
        self.I = I_indices
        self.markers = markers
        self.J = J_indices
        self.Product_Indices = Product_Indices
        self.M = len(I_indices)
        
        
    def computeScores(self,vector,k):
        self.k = min(self.M,k)
        self.loaded = True
        vec_i,vec_j,vec_prod = vector[self.I],vector[self.J], vector[self.Product_Indices]
        iaux, jaux = np.array(self.I-1), np.array(self.J-1)        
        li,lj,ui,uj = self.LB[iaux],self.LB[jaux],self.UB[iaux],self.UB[jaux]
        score1 = li*lj*vector[0] - lj*vec_i - li*vec_j + vec_prod
        score2 = -li*uj*vector[0] + uj*vec_i + li*vec_j - vec_prod
        score3 = -lj*ui*vector[0] + lj*vec_i + ui*vec_j - vec_prod
        score4 = ui*uj*vector[0] - uj*vec_i - ui*vec_j + vec_prod
        scores = np.concatenate([score1,score2,score3,score4])
        self.tuples = [(i, scores[i]) for i in range(4*self.M)]
        self.tuples.sort(key=itemgetter(1))
        self.temp = vector
        return [self.tuples[j][1] for j in range(self.k)]
    
    def retrieve(self,k2,only_neg=False):
        assert(self.loaded)
        k2 = min(k2,self.M)
        self.loaded = False
        assert(k2<=self.k)
        self.k = 0
        result = [] 
        res_markers = []
        for count in range(k2):
            if only_neg==False or self.tuples[count][1]<0:
                q = self.tuples[count][0]//self.M
                aux = self.tuples[count][0]%self.M
                i,j,edge = self.I[aux]-1,self.J[aux]-1, self.Product_Indices[aux]
                li,lj,ui,uj = self.LB[i],self.LB[j],self.UB[i],self.UB[j]
                x = [0, 1+i, 1+j, edge]
                assert(q<=3)
                if q==0:
                    coef = [li*lj,-lj,-li,1.0]
                if q==1:
                    coef = [-li*uj,uj,li,-1.0]
                if q==2:
                    coef = [-lj*ui,lj,ui,-1.0]
                if q ==3:
                    coef = [ui*uj,-uj,-ui,1.0]
                y = np.zeros(4)
                sg = coo_matrix((coef, (y, x)), shape=(1,self.N))
                assert(abs(sg.dot(self.temp)-self.tuples[count][1])<0.000001)
                result.append(sg.tocsr())
                res_markers.append(self.markers[aux])
        self.tuples = []
        return result, res_markers   

    def computeScores_and_retrieve(self,vector,k):
        scores = self.computeScores(vector,k)
        ps, markers = self.retrieve(len(scores))
        return scores, ps, markers
        

class TriangleOracle(Oracle):
    
    def __init__(self,N,edge_triplet_list):
        Oracle.__init__(self,N)
        self.edge_triplet_list = edge_triplet_list
        self.L = len(edge_triplet_list)
        self.A = [trip[0] for trip in edge_triplet_list]
        self.B = [trip[1] for trip in edge_triplet_list]
        self.C = [trip[2] for trip in edge_triplet_list]
        
    def computeScores(self,vector,k):
        self.k = min(4*self.L,k)
        self.loaded = True
        t0 = time.time()
        vec_a,vec_b,vec_c = vector[self.A],vector[self.B], vector[self.C]
        print("indices time = {0}".format(time.time()-t0))
        t0 = time.time()
        score1 = vector[0] + vec_a + vec_b + vec_c
        score2 = vector[0] + vec_a - vec_b - vec_c
        score3 = vector[0] - vec_a + vec_b - vec_c
        score4 = vector[0] - vec_a - vec_b + vec_c
        scores = np.concatenate([score1,score2,score3,score4])
        tuples = [(i, scores[i]) for i in range(4*self.L)]
        print("tuples time = {0}".format(time.time()-t0))
        print("len tuples = {0}".format(len(tuples)))
        t0 = time.time()
        self.tuples =  heapq.nsmallest(self.k,tuples,key = itemgetter(1))
        print("heap time = {0}".format(time.time()-t0))
        self.temp = vector
        return [self.tuples[j][1] for j in range(k)]
    
    def retrieve(self,k2,only_neg=False):
        assert(self.loaded)
        k2 = min(k2,self.k)
        self.loaded = False
        self.k = 0
        result = [] 
        res_markers = []
        for count in range(k2):
            if only_neg==False or self.tuples[count][1]<0:
                q = self.tuples[count][0]//self.L
                aux = self.tuples[count][0]%self.L
                a,b,c = self.A[aux],self.B[aux], self.C[aux]
                x = [0, a,b,c]
                if q == 0:
                    coef = [1,1,1,1]
                elif q ==1:
                    coef = [1,1,-1,-1]
                elif q ==2:
                    coef = [1,-1,1,-1]
                elif q ==3:
                    coef = [1,-1,-1,1]
                y = np.zeros(4)
                sg = coo_matrix((coef, (y, x)), shape=(1,self.N))
                assert(abs(sg.dot(self.temp)-self.tuples[count][1])<0.0000001)
                result.append(sg.tocsr())
                res_markers.append(0)
        self.tuples = []
        return result, res_markers   

    def computeScores_and_retrieve(self,vector,k):
        scores = self.computeScores(vector,k)
        ps, markers = self.retrieve(len(scores))
        return scores, ps, markers


class FiniteOracle(Oracle):
    
    def __init__(self,N,M,G, markers):
        Oracle.__init__(self,N)
        self.M = M
        self.matrix = G
        self.markers = markers
        assert(G.shape == (M,N))
        
    def computeScores(self,vector,k):
        self.k = min(self.M,k)
        self.loaded = True
        scores = self.matrix.dot(vector)
        self.tuples = [(i, scores[i]) for i in range(self.M)]
        self.tuples.sort(key=itemgetter(1))
        return [self.tuples[j][1] for j in range(self.k)]
    
    def retrieve(self,k2,only_neg = False):
        assert(self.loaded)
        k2 = min(k2,self.M)
        self.loaded = False
        assert(k2<=self.k)
        self.k = 0
        result = [(self.matrix.getrow(self.tuples[j][0])) for j in range(k2) if self.tuples[j][1]<0 or (only_neg==False) ] 
        res_markers = [self.markers[self.tuples[j][0]] for j in range(k2) if self.tuples[j][1]<0 or (only_neg==False)]
        self.tuples = []
        return result, res_markers

    def computeScores_and_retrieve(self,vector,k):
        scores = self.computeScores(vector,k)
        ps, markers = self.retrieve(len(scores))
        return scores, ps, markers
    
# class DenseSDPOracle(Oracle):
    
#     def __init__(self,N,matrixSize,x_positionsInSDP,y_positionsInSDP):
#         Oracle.__init__(self,N)
#         self.x_positionsInSDP = x_positionsInSDP
#         self.y_positionsInSDP = y_positionsInSDP
#         self.matrixSize = matrixSize
#         self.eigenvalues, self.eigenvectors = [],[]
#         assert(matrixSize*(matrixSize+1)/2==N)
    
#     def computeScores(self,vector,k):
#         self.loaded = True
#         self.k = min(self.matrixSize-1,k)
#         temp = coo_matrix((vector, (self.x_positionsInSDP, self.y_positionsInSDP)), shape=(self.matrixSize, self.matrixSize))
#         temp = temp.toarray()
#         temp = temp + temp.T - np.diag(temp.diagonal())
#         self.eigenvalues, self.eigenvectors = eigsh(temp, self.k, which = 'SA')
#         return [self.eigenvalues[j] for j in range(self.k)]
    
#     def retrieve(self,k2):
#         assert(self.loaded)
#         self.loaded = False
#         assert(k2<=self.k)
#         self.k = 0
#         res = []
#         for i in range(k2):
#             e = self.eigenvectors[:,i]
#             e = e.reshape((self.matrixSize,1))
#             fullmat = e.dot(e.T)
#             #print("Opération qui rallonge, à racourcir en utilisant un tenseur")
#             fullmat = fullmat + fullmat.T - np.diag(fullmat.diagonal())
#             res.append(csr_matrix(fullmat[self.x_positionsInSDP,self.y_positionsInSDP]))
#         self.eigenvalues, self.eigenvectors = [],[]
#         return res
    
class partialSDPOracle(Oracle):
    
    def __init__(self,N, matrixSize,vector_indices, x_submatrix_indices, y_submatrix_indices,marker):
        Oracle.__init__(self,N)
        self.vector_indices, self.x_positionsInSDP,self.y_positionsInSDP = vector_indices, x_submatrix_indices, y_submatrix_indices
        self.matrixSize = matrixSize
        self.eigenvalues, self.eigenvectors = [],[]
        assert(type(marker)==int)
        self.marker = marker
        
    def computeScores(self,vector,k):
        self.loaded = True
        self.k = min(int(0.5*self.matrixSize),k)
        self.k = min(self.k, MAX_EIGEN)
        vectorLimited = vector[self.vector_indices]
        temp = coo_matrix((vectorLimited, (self.x_positionsInSDP, self.y_positionsInSDP)), shape=(self.matrixSize, self.matrixSize))
        temp = temp.toarray()
        temp = temp + temp.T - np.diag(temp.diagonal())
        try:
            self.eigenvalues, self.eigenvectors = eigsh(temp, self.k, which = 'SA',tol = eigtol)
            return [self.eigenvalues[j] for j in range(self.k)]
        except:
            return [100 for j in range(self.k)]
    
    def retrieve(self,k2,only_neg=False):
        assert(self.loaded)
        self.loaded = False
        assert(k2<=self.k)
        self.k = 0
        m = len(self.x_positionsInSDP)
        res = []
        for i in range(len(self.eigenvalues)):
            if (self.eigenvalues[i]<0) or (only_neg==False):
                e = self.eigenvectors[:,i]
                e = e.reshape((self.matrixSize,1))
                fullmat = e.dot(e.T)
                #print("Opération qui rallonge, à racourcir en utilisant un tenseur")
                fullmat = fullmat + fullmat.T - np.diag(fullmat.diagonal())
                data = fullmat[self.x_positionsInSDP,self.y_positionsInSDP]
                vec = coo_matrix((data, (np.zeros(m),self.vector_indices)), shape=(1,self.N))
                res.append(vec.tocsr())
        self.eigenvalues, self.eigenvectors = [],[]
        return res, [self.marker]*len(res)
        
    def computeScores_and_retrieve(self,vector,k):
        scores = self.computeScores(vector,k)
        ps, markers = self.retrieve(len(scores))
        return scores, ps, markers

class ConstantOracle(Oracle):
    
    def __init__(self,N):
        self.N = N
        
    def computeScores(self,vector,k):
        return np.array([vector[0]])
    
    def retrieve(self,k, only_neg=False):
        vec = coo_matrix(([1], (np.zeros(1),np.zeros(1))), shape=(1,self.N))
        return vec, [0] #Arbitrary choice
        
    def computeScores_and_retrieve(self,vector,k):
        scores = self.computeScores(vector,k)
        ps, markers = self.retrieve(len(scores))
        return scores, ps, markers
    
class ListOracle(Oracle):
    
    def __init__(self,N,OracleList):
        Oracle.__init__(self,N)
        self.__oracles = OracleList
        for o in self.__oracles:
            assert(o.N==N)
            
    def addOracle(self,oracle):
        self.__oracles.append(oracle)
    
    def computeScores(self,vector,k):
        self.tuples = []
        self.loaded = True
        self.k = k
        oracleNumber = len(self.__oracles)
        for i in range(oracleNumber):
            temp = (self.__oracles[i]).computeScores(vector,k)
            klocal = min(k,len(temp))
            self.tuples.extend([(i,temp[j]) for j in range(klocal)])
        self.tuples.sort(key=itemgetter(1))
        self.tuples = self.tuples[:k]
        return [self.tuples[j][1] for j in range(k)]
        
    def retrieve(self,k2,only_neg=False):
        assert(self.loaded)
        self.loaded = False
        assert(k2<=self.k)
        self.k = 0
        podium = [self.tuples[j][0] for j in range(k2)]
        podium = Counter(podium).most_common()
        result = []
        res_markers = []
        for oracle_index, nb in podium:
            assert(nb<=k2)
            vectores, markers = (self.__oracles[oracle_index]).retrieve(nb,only_neg)
            result.extend(vectores)
            res_markers.extend(markers)
        self.tuples = []
        return result, res_markers

    def computeScores_and_retrieve(self,vector,k):
        scores = self.computeScores(vector,k)
        ps, markers = self.retrieve(len(scores))
        return scores, ps, markers

class MT_ListOracle(Oracle):
    #Class multithreaded
    
    def __init__(self,N,OracleList):
        Oracle.__init__(self,N)
        self.__oracles = OracleList
        for o in self.__oracles:
            assert(o.N==N)
    def non_decreasing(self,L):
        return all(x<=y for x, y in zip(L, L[1:]))
    def aux_call(self, tup):
        i, vector,k = tup
        return (self.__oracles[i]).computeScores_and_retrieve(vector,k)
    
    def addOracle(self,oracle):
        self.__oracles.append(oracle)
    
    def computeScores(self,vector,k):
        self.tuples = []
        self.loaded = True
        self.k = k
        oracleNumber = len(self.__oracles)
        result = [[]]*oracleNumber
        self.future_scores = [[]]*oracleNumber
        self.future_ps = [[]]*oracleNumber
        self.future_markers = [[]]*oracleNumber
        iterable = [(i,vector,k) for i in range(oracleNumber)]
        t0 = time.time()
        with Pool(processes=5) as pool: 
            result = pool.map_async(self.aux_call,iterable)
            pool.close()
            pool.join()
        print("duration = "+str(time.time()-t0))
        r=result.get()
        for i in range(oracleNumber):
            self.future_scores[i], self.future_ps[i],self.future_markers[i] = r[i]
            temp = self.future_scores[i]
            klocal = min(k,len(temp))
            self.tuples.extend([(i,temp[j]) for j in range(klocal)])
        
        self.tuples.sort(key=itemgetter(1))
        self.tuples = self.tuples[:k]
        return [self.tuples[j][1] for j in range(k)]
        
    def retrieve(self,k2):
        assert(self.loaded)
        self.loaded = False
        assert(k2<=self.k)
        self.k = 0
        podium = [self.tuples[j][0] for j in range(k2)]
        podium = Counter(podium).most_common()
        result = []
        res_markers = []
        for oracle_index, nb in podium:
            assert(nb<=k2)
            vectores, markers = self.__retrieve_aux(oracle_index,nb)
            result.extend(vectores)
            res_markers.extend(markers)
        self.tuples = []
        return result, res_markers
    
    def __retrieve_aux(self,oracle_index,nb):
        #print("remove assertion")
        #assert(self.non_decreasing(self.future_scores[oracle_index]))
        return self.future_ps[oracle_index][:nb],self.future_markers[oracle_index][:nb]
    