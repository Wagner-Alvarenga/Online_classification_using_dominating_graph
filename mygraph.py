#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 10:09:10 2021

@author: wagner
"""

import numpy as np
from scipy.spatial import distance_matrix

#-----------------------------------------------------------------------
# FUNCOES
#
# CalcMadj:         grafo de gabriel
# DSalgorithm:      conjunto dominante independente
# GGmold:           define indices e mascara p/ atualizacao grafo gabriel
# GGupdate:         atualiza X, Xb, D, Db utilizando funcao "UpdateD"
# DefineM:          determina matriz M grafo gabriel
#
#
#-----------------------------------------------------------------------

def CalcMadj(D,X):
    N = D.shape[0]
    n = X.shape[1]
    Adj_matrix = np.zeros((N,N))
    #auxcount = np.linspace(1,N,N//10).astype('int')
    for i in range(N):
        #if i in auxcount:
            #print(i,N)
        for j in range(N):
            if (i != j):
                d1 = (D[i,j])/2
                dist = np.array((X[i,:]+X[j,:])/2).reshape(n,1).T        
                d = distance_matrix(dist, X, p=2)      
                d[0,i] = float("inf")
                d[0,j] = float("inf")      
                compara = (d<d1)                                            
                if not compara.any():
                    Adj_matrix[i,j] = 1
                    Adj_matrix[j,i] = 1
    return Adj_matrix 


def DefineM(D,Db,Mask,N):
    Db2 = Db + Mask
    db_min = np.min(Db2,axis=0)
    d_2 = D[np.triu_indices(N, k=1)]/2
    m_tri = (d_2 <= db_min)*1
    M2 = np.zeros((N,N))
    ii =np.triu_indices(N,1)
    M2[ii[0],ii[1]] = m_tri 
    M2[ii[1],ii[0]] = m_tri
    return M2


def DSalgorithm(M,D,X,alpha,FlagExtrem):
    
    N = X.shape[0]
    _,ind_unique = np.unique(X,return_index=True,axis=0)
    Xind = np.linspace(0,N-1,N).astype(int)
    Xdupli = set(Xind) - set(ind_unique)

    # Calcula métrica
    MD = M*D
    Vsum = np.sum(M,axis=0)
    Dmean = np.sum(MD,axis=0)/Vsum

    Vmin = np.min(Vsum)
    Vmax = np.max(Vsum)
    Dmin = np.max(Dmean)
    Dmax = np.min(Dmean)
    
    N = Vsum.shape[0]
    
    testVequal = np.sum((Vsum[0] == Vsum[:])*1)
    if N != testVequal:
        Vnorm = (Vsum-Vmin)/(Vmax-Vmin)
    else:
        Vnorm = np.repeat(Vsum[0]/N,N)
        
    testDequal = np.sum((Dmean[0] == Dmean[:])*1)
    if N != testDequal:
        Dnorm = 1 - (Dmean-Dmin)/(Dmax-Dmin)
    else:
        Dnorm = np.repeat(Dmean[0]/N,N)
         
    metricaI = (Vnorm + (Dnorm*alpha))/2
    m = metricaI

    DP = np.empty([0])
    NDP = np.empty([0])
    
    if len(Xdupli) != 0:
        metricaI[list(Xdupli)] = 0
    
    # Faz pontos extremos como DP   (i = extremo    /  auxi = adjacente)  
    if FlagExtrem == 1:
        indext = np.where(Vsum==1)[0] 
        DP = np.hstack((DP,indext))
        Mext = M[indext,:]
        indext2 = np.sum(Mext,axis=0)
        indext2 = np.where(indext2>=1)[0]
        NDP = np.hstack((NDP,indext2))
        #for i in range(N):                                               
        #    auxi = np.nonzero(M[i,:])[0]                                     
        #    if auxi.shape[0] == 1:
        #        DP = np.hstack((DP,i))
        #        NDP = np.hstack((NDP,auxi))
        metricaI[DP.astype('int')] = 0
        metricaI[NDP.astype('int')] = 0

    DPi = np.where(metricaI==np.amax(metricaI))[0]     #[ind1 ind 2]
    if DPi.shape[0] > 1:
        Dnormall = Dnorm[DPi]
        Dnormind = np.where(Dnormall==np.amin(Dnormall))[0] 
        if Dnormind.shape[0] > 1:
            Vnormall = Vnorm[DPi]
            Vnormind = np.where(Vnormall==np.amax(Vnormall))[0] 
            DPi = DPi[Vnormind]
        else:
            DPi = DPi[Dnormind]
     
    NDPi = np.nonzero(M[DPi,:])[1] 
    DP = np.hstack((DP,DPi))
    NDP = np.hstack((NDP,NDPi))
   
    metricaI[DP.astype('int')] = 0
    metricaI[NDP.astype('int')] = 0
    
    while np.sum(metricaI) != 0:
        
        DPi = np.where(metricaI==np.amax(metricaI))[0]     #[ind1 ind 2]
        if DPi.shape[0] > 1:
            Dnormall = Dnorm[DPi]
            Dnormind = np.where(Dnormall==np.amin(Dnormall))[0] 
            if Dnormind.shape[0] > 1:
                Vnormall = Vnorm[DPi]
                Vnormind = np.where(Vnormall==np.amax(Vnormall))[0] 
                DPi = DPi[Vnormind]
            else:
                DPi = DPi[Dnormind]
       
        NDPi = np.nonzero(M[DPi,:])[1] 
        DP = np.hstack((DP,DPi))
        NDP = np.hstack((NDP,NDPi))
        
        metricaI[DP.astype('int')] = 0
        metricaI[NDP.astype('int')] = 0
        
    return np.sort(DP.astype('int')),np.sort(np.unique(NDP).astype('int')),Vnorm,Dnorm,m


def GGmold(z):
    Xpairs = np.array([[a,b] for a in range(z) for b in range(a,z) if a != b])
    indsize = Xpairs.shape[0]
    ind1 = np.tile(Xpairs[:,0],z).reshape(z,indsize)
    ind2 = np.tile(Xpairs[:,1],z).reshape(z,indsize)
    ind3 = np.tile(np.arange(z).T,indsize).reshape(z,indsize,order='F')
    test_i = (ind3==ind1)*1
    test_j = (ind3==ind2)*1
    Mask = test_i + test_j
    Mask = Mask * 999   #float('inf') 
    Indinsert = np.zeros(z-3)
    ii = 0
    for i in range(z-3):
        if i == 0:
            Indinsert[i] = z-2
        else:
            Indinsert[i] = Indinsert[i-1] + (z-2) - ii
            ii = ii + 1   
    Indinsert = np.hstack((Indinsert,Indinsert[-1]+2,Indinsert[-1]+3))
    indold = set(np.linspace(0,Xpairs.shape[0]-1,Xpairs.shape[0])) - set(Indinsert)

    return Xpairs, Mask, Indinsert.astype('int'),np.array(list(indold)).astype('int')

def GGupdate(xi,X,Xb,D,Db,N,n,indold,Indinsert):
    # D update
    di = distance_matrix(xi,X,p=2)
    D0 = UpdateD(D,di,N)

    # X update
    Xbsize = Xb.shape[0]
    X = X[1:,:]
    X0 = np.vstack((X,xi))

    # Xb update
    Xb0 = np.zeros((Xbsize,n))
    Xb_xi = (X0[:-1,:] + np.tile(xi,N-1).reshape(N-1,n))/2
    Xb0[Indinsert,:] = Xb_xi
    Xb0[indold,:] = Xb[N-1:,:]

    # Db update       
    Dbnew1 = distance_matrix(xi,Xb0,p=2)
    Dbnew2 = distance_matrix(Xb_xi,X0[:N-1,:],p=2)    
    sizeDb = Xb.shape[0]
    Db0 = np.zeros((N,sizeDb))
    Db0[:-1,indold] = Db[1:,N-1:]
    Db0[:-1,Indinsert] = Dbnew2.T
    Db0[-1,:] = Dbnew1
    return X0,Xb0,D0,Db0


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
