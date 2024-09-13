#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 10:09:10 2021

@author: wagner
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.approximation as nxaa
from scipy.spatial import distance_matrix

#-----------------------------------------------------------------------
# FUNCOES
#
# Domin:            dominante pacote networkx 
# ErroWin           calcula media movel 
# SplitXy:          divide treinamento e teste
# TrimNormaliza:    limita e normaliza dados [0,1] 
# KDEforplot:       calcula prob. kde para dados de plot
# KDEdist2          np.exp(-0.5*((Di**2)/(r**2)))
# KDEgauss:         calcula prob. kde para xi (h unico)    Di**2/r**r  
# KDElapla:         calcula prob. kde para xi (h unico)    Di   /r
# KDEiR:            calcula prob. kde para xi (h local)
# Normaliza:        normaliza dados [0,1] 
# PlotXDS:          plota janela e os dominantes 
# PlotGGXDS:        plota janela, dominantes e grafo
# UpdateD:          atualiza matriz de distancia
# UpdateWIN:        janela movel
#
#-----------------------------------------------------------------------


def Domin(Adj_matrix):
    G = nx.Graph()
    G = nx.from_numpy_matrix(Adj_matrix)
    Dset = nxaa.min_weighted_dominating_set(G, weight=None)  
    return  Dset

def ErroWin(errote,Wlen):
    erroPad = np.concatenate((np.zeros(Wlen).reshape(Wlen,1),errote), axis=0)
    Nerro = errote.shape[0]
    Werro = np.zeros(Nerro-Wlen)
    for i in range(Nerro-Wlen):
        ii = i + Wlen
        Werro[i] = np.sum(erroPad[(ii-Wlen):ii])/Wlen
    return Werro

def SplitXy(X,y,Ncut,N):
    xtrain = np.array(X[0:Ncut,:])
    ytrain = np.array(y[0:Ncut,:])               
    xtest = np.array(X[(Ncut+1):N,:])
    ytest = np.array(y[(Ncut+1):N,:])            
    return xtrain, ytrain, xtest, ytest

def TrimNormaliza(X,Xmin,Xmax): 
    n = X.shape[1]
    for i in range(n):
        X[X[:,i] < Xmin[i]] = Xmin[i]
        X[X[:,i] > Xmax[i]] = Xmax[i]                       
    X = (X-Xmin)/(Xmax-Xmin)
    return X

def Normaliza(X): 
    Xmin = np.min(X,axis=0)
    Xmax = np.max(X,axis=0)                  
    X = (X-Xmin)/(Xmax-Xmin)
    return X

def KDEforplot(C,X,r,n): 
    N = X.shape[0] 
    Csize = C.shape[0] 
    p = np.zeros(N) 
    for i in range(N): 
        xi = X[i,:].reshape(1,2) 
        Di = distance_matrix(xi,C,p=2)  
        aux = np.exp(-0.5*((Di**2)/(r**2)))  
        p[i] =  np.sum( aux / (r**n) ) / Csize 
    return p 

def KDEdist2(C,xi,r,n): 
    Di = distance_matrix(xi,C,p=2)  
    aux = (r**n) * C.shape[0] 
    p = np.sum(  np.exp(-0.5*((Di**2)/(r**2))))  /  aux
    return p 


def KDEgauss(C,xi,r,n): 
    Di = distance_matrix(xi,C,p=2)  
    p = np.sum(  np.exp(-0.5*((Di**2)/(r**2))))  /  ( (r**n) * C.shape[0] )  
    return p 

def KDElapla(C,xi,r,n): 
    Di = distance_matrix(xi,C,p=2)  
    p = np.sum(  np.exp(Di/r))  /  ( (r**n) * C.shape[0] )  
    return p 

def KDEiR(C,xi,R,n): 
    Di = distance_matrix(xi,C,p=2)  
    p = np.sum(  np.exp( -0.5*((Di**2)/(R**2)))  / (R**n) )  /  (C.shape[0])  
    return p 

def PlotXDS(X,DP,xmin,xmax,ymin,ymax):
    #xmin = np.min(X[:,0]) - 0.5
    #xmax = np.max(X[:,0]) + 0.5
    #ymin = np.min(X[:,1]) - 0.5
    #ymax = np.max(X[:,1]) + 0.5
    fig,(p1,p2,p3) = plt.subplots(1,3, figsize=(15,4))
    p1.plot(X[:,0],X[:,1],'.k', label='$V$')
    p1.axis([xmin,xmax,ymin,ymax])
    p2.plot(DP[:,0],DP[:,1],'.r', label='$S$')
    p2.axis([xmin,xmax,ymin,ymax])
    p3.plot(X[:,0],X[:,1],'.k', label='$V-S$')
    p3.plot(DP[:,0],DP[:,1],'.r', label='$S$')
    p3.axis([xmin,xmax,ymin,ymax])
    #p1.set_title('Todos os pontos')
    #p2.set_title('Dominantes')
    p1.legend()
    p2.legend()
    p3.legend()
    plt.show()
    
def PlotGGXDS(X,XD,M):
    x1min = np.min(X[:,0])
    x1max = np.max(X[:,0])
    x2min = np.min(X[:,1])
    x2max = np.max(X[:,1])
    deltx1 = (x1max - x1min) * 0.1
    deltx2 = (x2max - x2min) * 0.1
    x1min = x1min - deltx1 
    x1max = x1max + deltx1
    x2min = x2min - deltx2
    x2max = x2max + deltx2
    
    fig, p1 = plt.subplots(1,1, figsize=(4,4)) 
    p1.scatter(X[:,0],X[:,1],c='b')
    p1.scatter(XD[:,0],XD[:,1],c='r')
    for i in range(M.shape[0]):
        adj = np.argwhere(M[:,i]==1)
        x1 = np.array(X[i,:]).reshape(1,2)
    
        for j in range(adj.shape[0]):
            x2 = np.array(X[adj[j,0],:]).reshape(1,2)   
            x1x2 = np.concatenate((x1,x2),axis=0)
            p1.plot(x1x2[:,0],x1x2[:,1], c='k',lw=0.7)
    p1.axis([x1min,x1max,x2min,x2max])
    p1.set_xticklabels([])
    p1.set_yticklabels([]) 
    plt.show()

def UpdateD(D,di,sizeW):
    sizenow = D.shape[0]
    #print("D",sizenow)
    #print("di", di.shape[1])
    Dnew = np.vstack((D,di))
    if sizenow == sizeW:
        Dnew2 = np.hstack((Dnew,np.hstack((di,np.array(0).reshape(1,1))).T))
        Dnew3 = Dnew2[1:Dnew2.shape[0],1:Dnew2.shape[1]]
    elif sizenow > sizeW:
        ncut = sizenow - sizeW
        Dnew3 = Dnew[ncut:sizenow,ncut:sizenow]
    else:
        Dnew3 = np.hstack((Dnew,np.hstack((di,np.array(0).reshape(1,1))).T))
    return Dnew3

def UpdateWIN(Win,samp,Wsize) :
    if Win.shape[0] < Wsize:                                                         
        Win = np.vstack((Win,samp))
    else:
        Dsamps = 1
        Win = np.delete(Win,[np.arange(0,Dsamps)], axis=0)              
        Win = np.vstack((Win,samp))
    return Win
