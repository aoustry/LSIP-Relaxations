# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:25:09 2020

@author: aoust
"""
import numpy as np
import random
from scipy.sparse import vstack
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.sparse import diags

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

class Bundle():
    
    def __init__(self, N,first_abs = 1,maintainGramMatrix  = False, pruningThreshold = 1.0E-7, compressed_size = 10000):
        self.N = N
        self.M = 0
        self.maintainGramMatrix = maintainGramMatrix
        assert(maintainGramMatrix==False)
        self.data = []
        self.weights = []
        self.markers = []
        if self.maintainGramMatrix:
            self.__GramMatrix = np.zeros((0,0))
        self.threshold = pruningThreshold
        self.first_abs = first_abs
        self.compressed_size = compressed_size
    
    def __str__(self):
        return "Bundle : Number of elements {0}, sum of weights {1}".format(len(self.data), sum(self.weights))
        
    def aggregation(self):
        #return (np.array(self.weights)).dot(np.array(self.data))
        return (self.data.T).dot(np.array(self.weights))
        #data (M,N)
        
        
    def add(self,new_vectors,markers,new_weights=None):
        #ASSERTION_MARK
        if new_weights==None:
            new_weights =  [0 for i in range(len(new_vectors))]
        else:
            assert(len(new_weights)==len(new_vectors))
        for v in new_vectors:
            assert((v.shape==(self.N,)) or v.shape==(1,self.N))
            #PROBLEM WITH data
            
        if self.maintainGramMatrix:
            if self.M>0:
                csr_new= (vstack(new_vectors).tocsr(copy=True)).transpose()
                extract = csr_new[self.first_abs:,:]
                B = self.data[:,self.first_abs:].dot(extract)
                D = (extract.transpose(copy=True)).dot(extract)
                self.__GramMatrix = hstack((self.__GramMatrix,B))
                temp = hstack((B.transpose(),D))
                self.__GramMatrix = vstack((self.__GramMatrix,temp)).tocsr()
                
            else:
                csr_new= (vstack(new_vectors).tocsr(copy=True))
                self.__GramMatrix = (csr_new[:,self.first_abs:]).dot((csr_new[:,self.first_abs:]).transpose()).tocsr()
            
       
        if self.M>0:
            self.data= vstack([self.data]+ new_vectors)
            self.weights.extend(new_weights)
            self.markers.extend([m for m in markers])
            self.M = self.M + len(new_vectors)
        else:
            self.data = vstack(new_vectors)
            self.M = len(new_vectors)
            self.weights = new_weights
            self.markers = [m for m in markers]
            
    def multiplicativeWeightUpdate(self,alpha):
        assert(alpha>=0)
        assert(alpha<=1)
        self.weights = [alpha* w for w in self.weights]
    
    def updateWeights(self,vec):
        assert(vec.shape==(self.data.shape[0],))
        sumofweights = sum(self.weights)
        for i in vec:
            assert(i>=-max(1,sumofweights/1000))
        self.weights = list(vec)
    
    def prune(self):
        print("Pruning bundle - Length before pruning {0}".format(self.M))
        indices = [i for i in range(self.M) if self.weights[i]>self.threshold]
        self.data = self.data[indices,:]
        self.weights = list(np.array(self.weights)[indices])
        self.markers = list(np.array(self.markers)[indices])
        self.M = len(indices)
       
        # self.data = [self.data[i] for i in range(M) if self.weights[i]>self.threshold]
        if self.maintainGramMatrix:
            self.__GramMatrix = self.__GramMatrix[indices][:,indices]
        # self.weights = [self.weights[i] for i in range(M) if self.weights[i]>self.threshold]
        # assert(len(self.weights)==len(self.data))
        # assert(len(self.weights)==len(self.__GramMatrix))
        print("Pruning bundle - Length after pruning {0}".format(self.M))
        
  
    def compress(self):
        print("Bundle compression...")
        markers_modulo_k = [m%self.compressed_size for m in self.markers]
        new_markers = list(set(markers_modulo_k))
        new_markers.sort()
        new_size = len(new_markers)
        reverse_new_markers = {new_markers[i] : i for i in range(new_size)}
        
        transfer = np.zeros((new_size,self.M))
        for j in range(self.M):
            i = reverse_new_markers[markers_modulo_k[j]]
            transfer[i,j] = self.weights[j]
        new_weights = transfer.sum(axis=1)
        for i in range(new_size):
            transfer[i,:] = transfer[i,:]/new_weights[i]
        transfer = csr_matrix(transfer)
        self.data = transfer.dot(self.data)
        self.weights = list(new_weights)
        self.markers = new_markers
        self.M =new_size
        del transfer
        
        if self.maintainGramMatrix:
             print("Computing the new Grammian")
             self.__GramMatrix = (self.data[:,self.first_abs:]).dot((self.data[:,self.first_abs:]).transpose(copy=True))
        print("Bundle compression finished")
    
   
         
    def compressInK_deprecated(self,k):
        print("Compressing bundle - Warning : random fonction")
        assert(k>1)
        self.prune()
        transfer = np.zeros((k,self.M))
        self.weights = [max(w,0) for w in self.weights]
        for j in range(self.M):
            i = random.randint(0,k-1)
            transfer[i,j] = self.weights[j]
        new_weights = transfer.sum(axis=1)
        assert(len(new_weights)==k)
        for i in range(k):
            transfer[i,:] = transfer[i,:]/new_weights[i]
        transfer = csr_matrix(transfer)
        self.data = transfer.dot(self.data)
        self.weights = list(new_weights)
        self.M =k
        del transfer
        if self.maintainGramMatrix:
             
             self.__GramMatrix = (self.data[:,self.first_abs:]).dot((self.data[:,self.first_abs:]).transpose(copy=True))
            
            
    def dot(self,array):
        #ASSERT_MARK
        if self.first_abs == 1:
            assert(array[0]==1)
        return self.data.dot(array)
    
    def Gram(self, dense = False):
        
        if self.maintainGramMatrix:
            print("Gram Matrix Shape = {0}".format(self.__GramMatrix.shape))
            return self.__GramMatrix 
        else:
            if self.first_abs==1:
                if dense:
                    copy = self.data.toarray()
                    copy[:,0] = np.zeros(self.M)
                    return copy.dot(copy.T)
                else:
                    copy = csr_matrix(self.data)
                    vec = np.ones(self.N)
                    vec[0] = 0
                    mask  = diags(vec)
                    copy = copy.dot(mask)
                    return copy.dot(self.data.transpose(copy=True))
            else:
                assert(self.first_abs==0)
                if dense:
                    copy = self.data.toarray()
                    return copy.dot(copy.T)
                else:
                    return self.data.dot(self.data.transpose(copy=True))
                    
                
                    
            
            #return np.array([[(a[self.first_abs:]).dot(b[self.first_abs:]) for a in self.data] for b in self.data])