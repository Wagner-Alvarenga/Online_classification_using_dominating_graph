#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 10:08:36 2021

@author: wagner
"""

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

#-----------------------------------------------------------------------
# FUNCOES
#
# X, y = Gaussians4dMov(777)                         # 5.000
# X, y = TwoGaussIncre(777)                          # 5.000
# X, y = CheckerRotIncre(777)                        # 5.000
# X, y = TwoMoonsRot(777)                            # 5.000
# X, y = SpiralsRot(777)                             # 5.000

# X,y,TrueDrift = SpiralsAbruptRot(777,10000)       # 10.000
# X,y,TrueDrift = CheckerAbruptRot(777,10000)       # 10.000
# X,y,TrueDrift = TwoGaussAbruptRot(777,10000)      # 10.000
# X,y,TrueDrift = TwoMoonsAbruptRot(777,10000)      # 10.000
#
#-----------------------------------------------------------------------

    

def Gaussians4dMov(seednum):
    np.random.seed(seednum)
    N = 20000
    N = N//8
    
    # Dataset dividido em 2 trechos, com mudancas incrementais:
    #   No 1o classe 3 troca de lugar com classe 4 
    #   No 2o classe 1 troca de lugar com classe 2
    # 4 dimensoes - n
    # 3 classes
    # 2 centros p\ classe - c
    n = 4
    c = 2
    
    c11 = np.array([-6.  ,  0. , -16.5 ,-22.5])   
    c12 = np.array([-16.5 ,-21. ,  -7.5 , -7.5]) 
    
    c21 = np.array([-3. , -16.5 ,-19.5 , -9.])  
    c22 = np.array([-25.5 , -6.  , -6. , -16.5])   
    
    c31 = np.array([-7.5 ,-10.5  , 1.5, -10.5])  
    c32 = np.array([-13.5 ,-15. ,  -3. , -15. ])   
    
    c41 = np.array([1.5 , -7.5 ,-22.5 ,-25.5])   
    c42 = np.array([-27.,  -24. , -10.5  , 3.])   
    
    aux0 = np.min([np.min(c11),
                np.min(c12),
                np.min(c21),
                np.min(c22),
                np.min(c31),
                np.min(c32),
                np.min(c41),
                np.min(c42)])

    aux0 = np.repeat(aux0,n)*(-1)         # se alterar os centros, mude aqui!
    c11 = c11 + aux0
    c12 = c12 + aux0
    c21 = c21 + aux0
    c22 = c22 + aux0
    c31 = c31 + aux0
    c32 = c32 + aux0
    c41 = c41 + aux0
    c42 = c42 + aux0
    
    delt11 = c31 - c41
    delt12 = c32 - c42
    delt21 = c11 - c21
    delt22 = c12 - c22
    
    delt11 = delt11/(N/2)
    delt12 = delt12/(N/2)
    delt21 = delt21/(N/2)
    delt22 = delt22/(N/2)
    
    X = np.empty([0,n])
    y = np.empty([0,1])
    
    X1 = np.empty([0,n])
    X2 = np.empty([0,n])
    X3 = np.empty([0,n])
    X4 = np.empty([0,n])
    
    y1 = np.empty([0,1])
    y2 = np.empty([0,1])
    y3 = np.empty([0,1])
    y4 = np.empty([0,1])
    
    
    for i in range(N):
        
        x11 = np.array(np.random.randn(1,n) + c11).reshape(1,n)
        x12 = np.array(np.random.randn(1,n) + c12).reshape(1,n)
        x21 = np.array(np.random.randn(1,n) + c21).reshape(1,n)
        x22 = np.array(np.random.randn(1,n) + c22).reshape(1,n)   
        x31 = np.array(np.random.randn(1,n) + c31).reshape(1,n)
        x32 = np.array(np.random.randn(1,n) + c32).reshape(1,n)
        x41 = np.array(np.random.randn(1,n) + c41).reshape(1,n)
        x42 = np.array(np.random.randn(1,n) + c42).reshape(1,n)
        
        X1 = np.vstack((X1,x11))
        X1 = np.vstack((X1,x12))
        y1 = np.vstack((y1,np.repeat(1,c).reshape(c,1)))
        
        X2 = np.vstack((X2,x21))
        X2 = np.vstack((X2,x22))
        y2 = np.vstack((y2,np.repeat(2,c).reshape(c,1)))
        
        X3 = np.vstack((X3,x31))
        X3 = np.vstack((X3,x32))
        y3 = np.vstack((y3,np.repeat(3,c).reshape(c,1)))
        
        X4 = np.vstack((X4,x41))
        X4 = np.vstack((X4,x42))
        y4 = np.vstack((y4,np.repeat(4,c).reshape(c,1)))
        
        X = np.vstack((X1,X2,X3,X4))
        y = np.vstack((y1,y2,y3,y4))
        
        if i < N//2:
    
            c31 = c31 - delt11
            c32 = c32 - delt12
            c41 = c41 - delt11*(-1)
            c42 = c42 - delt12*(-1)
            
        else:
            
            c11 = c11 - delt21
            c12 = c12 - delt22
            c21 = c21 - delt21*(-1)
            c22 = c22 - delt22*(-1)
    
    #plt.scatter(X[:,0],X[:,1],c=y)
    #plt.show()
    y = y.astype('int')
    y = y.reshape(y.shape[0],1)
    return X, y   

def TwoGaussIncre(seednum): 
    N = 5000
    #seednum = 777
    np.random.seed(777)
    mean1 = np.array([-5, 0])
    mean2 = np.array([5, 0])
    cov = [[10, 0], [0, 10]]
    y = np.zeros(N)
    X = np.zeros((N,2))
    
    delt = 10/N
    
    for i in range(0,N):
         
        randi = np.random.rand(1)

        if randi < 0.5:
            x1, x2 = np.random.multivariate_normal(mean1, cov, 1).T
            y[i] = 1
            X[i,:] = np.hstack((x1,x2))   
        else:
            x1, x2 = np.random.multivariate_normal(mean2, cov, 1).T
            y[i] = -1
            X[i,:] = np.hstack((x1,x2))
            
        if i < N//2:
            mean1 = mean1 + np.array([delt,0])
            
        else:
            mean1 = mean1 - np.array([delt,0])
        
    Xmax = np.max(X, axis=0) 
    Xmin = np.min(X, axis=0)  
    X = (X - Xmin)/(Xmax - Xmin + 0.000000001)  -0.5   
     
    #plt.scatter(X[:,0],X[:,1],c=y)
    #plt.show()
    y = y.astype('int')
    y = y.reshape(y.shape[0],1)
    return X, y     

def CheckerRotIncre(seednum): 
    N = 5010
    #seednum = 777
    np.random.seed(seednum)
    #N = N + 500
    Ni = N//16
    n = 2
    delt1 = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3])
    delt2 = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])
    yi =np.array([1,0,1,0,
                  0,1,0,1,
                  1,0,1,0,
                  0,1,0,1])
    X = np.empty([0,n])
    y = np.empty([0])
    for i in np.arange(0,16): 
        x11 = np.random.rand(Ni,1) + delt1[i]    
        x12 = np.random.rand(Ni,1) + delt2[i]
        X1 = np.concatenate((x11,x12), axis=1)
        y1 = np.repeat(yi[i],Ni)
        X = np.vstack((X,X1))  
        y = np.hstack((y,y1))  
    db = np.hstack((X,y.reshape(y.shape[0],1)))
    ind = np.random.permutation(db.shape[0]) 
    ind = np.random.permutation(ind) 
    db = np.array(db[ind,:])  
    
    X = np.array(db[:,0:2])
    y[y==0]=-1
    y = np.array(db[:,2]).astype('int')
    
    Xmax = np.max(X, axis=0) 
    Xmin = np.min(X, axis=0)  
    X = (X - Xmin)/(Xmax - Xmin + 0.000000001)  -0.5
    X[:,0] = X[:,0] + 0.5
    X[:,1] = X[:,1] + 0.5
    
    tetha = np.pi/4  
    delttetha = tetha/N 
    Mrot = np.array([np.cos(delttetha),
                     np.sin(delttetha),
                     -np.sin(delttetha),
                     np.cos(delttetha)]).reshape(2,2)
    Xaux = np.empty([0,2])
     
    #plt.scatter(X[:,0],X[:,1],c=y)
    #plt.show()
    
    for i in range(5000):
        xi = np.array(X[i:(i+1),:]).reshape(1,2)
        xirot = xi @ Mrot
        Xaux = np.vstack((Xaux,xirot))
        Mrot = np.array([np.cos(delttetha*(i+1)),
                     np.sin(delttetha*(i+1)),
                     -np.sin(delttetha*(i+1)),
                     np.cos(delttetha*(i+1))]).reshape(2,2)
     
    Xaux = np.array(Xaux[0:5000,:])
    y = np.array(y[0:5000])
    y = y.reshape(y.shape[0],1)
    y = y.astype('int')
    return Xaux, y

def TwoMoonsRot(seednum): 
    d1 = 5000
    np.random.seed(seednum)
    N1 = np.round(d1*0.8).astype('int')
    N2 = d1 - N1
    X1,y1 = make_moons(n_samples=N1, shuffle=True, noise=0.15, random_state=None)
    X2,y2 = make_moons(n_samples=N2, shuffle=True, noise=0.15, random_state=None)
    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2))
    ind = np.random.permutation(np.arange(d1))
    X = np.array(X[ind,:])                                           
    y = np.array(y[ind]) 
    #indy1 = np.where(y==1)[0]
    #X = X[indy1,:]
    #y = y[indy1]
    
    Xmax = np.max(X, axis=0) 
    Xmin = np.min(X, axis=0)  
    X = (X - Xmin)/(Xmax - Xmin + 0.000000001)  -0.5 
    
    repete = X.shape[0]
    n = X.shape[1]
    deltrot = (2*np.pi/repete)  
    X2 = np.empty([0,n])
    for i in range(repete):
        xi = X[i,:].reshape(1,2)
        tetha = deltrot * i
        Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
        xi = np.array(xi @ Mrot).reshape(1,2)
        X2 = np.vstack((X2,xi))
    
    X = X2
      
    #plt.scatter(X[:,0],X[:,1],c=y)
    #plt.show()
    
    y[y==0] = -1 
    y = y.astype('int')
    y = y.reshape(y.shape[0],1)
    return X,y 


def SpiralsRot(seednum): 
    n_points = 5000
    np.random.seed(seednum)
    noise=.5
    n_points = np.round(n_points/2).astype('int')
    n = np.sqrt(np.random.rand(n_points,1)) * 720 * (2*np.pi)/360   # 720
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    X1 = np.hstack((d1x,d1y))
    X2 = np.hstack((-d1x,-d1y))
    y1 = np.zeros(n_points)
    y2 = np.ones(n_points)
    X2 = X2 + np.random.randn(n_points, 2)*0.3
    X1 = X1 + np.random.randn(n_points, 2)*0.3
    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2))
    ind = np.random.permutation(np.arange(n_points*2))
    X = np.array(X[ind,:])                                           
    y = np.array(y[ind])   
    Xmax = np.max(X, axis=0) 
    Xmin = np.min(X, axis=0)  
    X = (X - Xmin)/(Xmax - Xmin + 0.000000001)  -0.5 
    
    repete = X.shape[0]
    n = X.shape[1]
    deltrot = (np.pi)/repete  
    X2 = np.empty([0,n])
    for i in range(repete):
        xi = X[i,:].reshape(1,2)
        tetha = deltrot * i
        Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
        xi = np.array(xi @ Mrot).reshape(1,2)
        X2 = np.vstack((X2,xi))
    
    X = X2
     
    #plt.scatter(X[:,0],X[:,1],c=y)
    #plt.show()
    
    y[y==0] = -1 
    y = y.astype('int')
    y = y.reshape(y.shape[0],1)
    return X,y 


def SpiralsAbruptRot(seednum,n_points):
    Nsize = 0
    np.random.seed(seednum)
    d4 = (n_points-Nsize)//4
    noise=.5
    n_points = np.round(n_points/2).astype('int')
    n = np.sqrt(np.random.rand(n_points,1)) * 720 * (2*np.pi)/360   # 720
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    X1 = np.hstack((d1x,d1y))
    X2 = np.hstack((-d1x,-d1y))
    y1 = np.zeros(n_points)
    y2 = np.ones(n_points)
    X2 = X2 + np.random.randn(n_points, 2)*0.3
    X1 = X1 + np.random.randn(n_points, 2)*0.3
    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2))
    ind = np.random.permutation(np.arange(n_points*2))
    X = np.array(X[ind,:])                                           
    y = np.array(y[ind])   
     
    X1 = np.array(X[0:(Nsize+(d4)*1),:])
    y1 = np.array(y[0:(Nsize+(d4)*1)])
    X2 = np.array(X[(Nsize+(d4)*1):(Nsize+(d4)*2),:])
    y2 = np.array(y[(Nsize+(d4)*1):(Nsize+(d4)*2)])
    X3 = np.array(X[(Nsize+(d4)*2):(Nsize+(d4)*3),:])
    y3 = np.array(y[(Nsize+(d4)*2):(Nsize+(d4)*3)])
    X4 = np.array(X[(Nsize+(d4)*3):(Nsize+(d4)*4),:])
    y4 = np.array(y[(Nsize+(d4)*3):(Nsize+(d4)*4)])
    
    tetha = 1.6
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X2 = np.array(X2[:,0:2]) @ Mrot
    tetha = 3.2
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X3 = np.array(X3[:,0:2]) @ Mrot
    tetha = 4.8
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X4 = np.array(X4[:,0:2]) @ Mrot
    
    ##fig, (p1,p2,p3,p4) = plt.subplots(1, 4, figsize=(20, 4))
    #fig, (p1,p2,p3,p4) = plt.subplots(1, 4, figsize=(20, 4))
    #p1.scatter(X1[:,0],X1[:,1], c=y1)
    #p1.axis([-15, 15, -15, 15])  
    #p2.scatter(X2[:,0],X2[:,1], c=y2)
    #p2.axis([-15, 15, -15, 15])  
    #p3.scatter(X3[:,0],X3[:,1], c=y3)
    #p3.axis([-15, 15, -15, 15])  
    #p4.scatter(X4[:,0],X4[:,1], c=y4)
    #p4.axis([-15, 15, -15, 15])  
    #fig.suptitle('Spirals', fontsize=25,fontweight='bold')
    #fig.show()
    #plt.show()
       
    X = np.vstack((X1,X2,X3,X4))  
    y = np.hstack((y1,y2,y3,y4)) 
                                        
    y[y==0] = -1 
    y = y.astype('int')
    y = y.reshape(y.shape[0],1)
    TrueDrift = np.zeros(X.shape[0]) - 1
    TrueDrift[2500] = 1
    TrueDrift[5000] = 1
    TrueDrift[7500] = 1
    return X,y,TrueDrift

def CheckerAbruptRot(seednum,N): 
    Nsize = 0
    np.random.seed(seednum)
    Ni = (N+Nsize)//16
    d4 = N//4
    n = 2
    delt1 = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3])
    delt2 = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])
    yi =np.array([1,0,1,0,
                  0,1,0,1,
                  1,0,1,0,
                  0,1,0,1])
    X = np.empty([0,n])
    y = np.empty([0])
    for i in np.arange(0,16): 
        x11 = np.random.rand(Ni,1) + delt1[i]    
        x12 = np.random.rand(Ni,1) + delt2[i]
        X1 = np.concatenate((x11,x12), axis=1)
        y1 = np.repeat(yi[i],Ni)
        X = np.vstack((X,X1))  
        y = np.hstack((y,y1))  
    db = np.hstack((X,y.reshape(y.shape[0],1)))
    ind = np.random.permutation(db.shape[0]) 
    ind = np.random.permutation(ind) 
    db = np.array(db[ind,:])  
    
    X = np.array(db[:,0:2])
    y = np.array(db[:,2])
    
    Xmax = np.max(X, axis=0) 
    Xmin = np.min(X, axis=0)  
    X = (X - Xmin)/(Xmax - Xmin + 0.000000001)  -0.5

    X1 = np.array(X[0:(Nsize+(d4)*1),:])
    y1 = np.array(y[0:(Nsize+(d4)*1)])
    X2 = np.array(X[(Nsize+(d4)*1):(Nsize+(d4)*2),:])
    y2 = np.array(y[(Nsize+(d4)*1):(Nsize+(d4)*2)])
    X3 = np.array(X[(Nsize+(d4)*2):(Nsize+(d4)*3),:])
    y3 = np.array(y[(Nsize+(d4)*2):(Nsize+(d4)*3)])
    X4 = np.array(X[(Nsize+(d4)*3):(Nsize+(d4)*4),:])
    y4 = np.array(y[(Nsize+(d4)*3):(Nsize+(d4)*4)])
    
    tetha = 0.8 #0.5
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X2 = np.array(X2[:,0:2]) @ Mrot
    tetha = 1.6 #1
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X3 = np.array(X3[:,0:2]) @ Mrot
    tetha = 2.4 #1.6
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X4 = np.array(X4[:,0:2]) @ Mrot
    
    #fig, (p1,p2,p3,p4) = plt.subplots(1, 4, figsize=(20, 4))
    #p1.scatter(X1[:,0],X1[:,1], c=y1)
    #p1.axis([-0.75, 0.75, -0.75, 0.75])  
    #p2.scatter(X2[:,0],X2[:,1], c=y2)
    #p2.axis([-0.75, 0.75, -0.75, 0.75]) 
    #p3.scatter(X3[:,0],X3[:,1], c=y3)
    #p3.axis([-0.75, 0.75, -0.75, 0.75]) 
    #p4.scatter(X4[:,0],X4[:,1], c=y4)
    #p4.axis([-0.75, 0.75, -0.75, 0.75]) 
    #fig.suptitle('Checkerboard', fontsize=25,fontweight='bold')
    #fig.show()
    #plt.show()
    
    X = np.vstack((X1,X2,X3,X4))  
    y = np.hstack((y1,y2,y3,y4)) 
    y = y.astype('int')
    y = y.reshape(y.shape[0],1)
    y[y==0] = -1 
    TrueDrift = np.zeros(X.shape[0]) - 1
    TrueDrift[2500] = 1
    TrueDrift[5000] = 1
    TrueDrift[7500] = 1
    return X,y,TrueDrift


def TwoGaussAbruptRot(seednum,N):
    Nsize = 0
    np.random.seed(seednum)
    d4 = (N-Nsize)//4
    np.random.seed(777)
    mean1 = np.array([-5.5, 0])
    mean2 = np.array([5.5, 0])
    cov = [[10, 0], [0, 10]]
    y = np.zeros(N)
    X = np.zeros((N,2))
    
    for i in range(0,N):
        
        randi = np.random.rand(1)

        if randi < 0.5:
            x1, x2 = np.random.multivariate_normal(mean1, cov, 1).T
            y[i] = 1
            X[i,:] = np.hstack((x1,x2))   
        else:
            x1, x2 = np.random.multivariate_normal(mean2, cov, 1).T
            y[i] = -1
            X[i,:] = np.hstack((x1,x2))
            
  
    # rotaciona   
    tetha = 1.6
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X2 = np.array(X[(Nsize+(d4)*1):(Nsize+(d4)*2),:]) @ Mrot
    
    tetha = 3.15
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X3 = np.array(X[(Nsize+(d4)*2):(Nsize+(d4)*3),:]) @ Mrot
    
    tetha = 4.7
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X4 = np.array(X[(Nsize+(d4)*3):(Nsize+(d4)*4),:]) @ Mrot
    
    #fig, (p1,p2,p3,p4) = plt.subplots(1, 4, figsize=(20, 4))
    #p1.scatter(X[0:(Nsize+(d4)*1),0],X[0:(Nsize+(d4)*1),1], c=y[0:(Nsize+(d4)*1)])
    #p1.axis([-15, 15, -15, 15])  
    #p2.scatter(X2[:,0],X2[:,1], c=y[(Nsize+(d4)*1):(Nsize+(d4)*2)])
    #p2.axis([-15, 15, -15, 15])  
    #p3.scatter(X3[:,0],X3[:,1], c=y[(Nsize+(d4)*2):(Nsize+(d4)*3)])
    #p3.axis([-15, 15, -15, 15])  
    #p4.scatter(X4[:,0],X4[:,1], c=y[(Nsize+(d4)*3):(Nsize+(d4)*4)])
    #p4.axis([-15, 15, -15, 15])  
    #fig.suptitle('Gaussians', fontsize=25,fontweight='bold')
    #fig.show()
    #plt.show()
    
    X = np.vstack((X[0:(Nsize+(d4)*1),:],X2,X3,X4))
    y[y==0] = -1 
    y = y.astype('int')
    #y = np.hstack((y[0:1100],y[1000:N]))
    y = y.reshape(y.shape[0],1)
    TrueDrift = np.zeros(N) - 1
    TrueDrift[2500] = 1
    TrueDrift[5000] = 1
    TrueDrift[7500] = 1
    # TrueDrift[Nsize+(d4)*1] = 1
    # TrueDrift[Nsize+(d4)*2] = 1
    # TrueDrift[Nsize+(d4)*3] = 1
    return X,y,TrueDrift


def TwoMoonsAbruptRot(seednum,d1):  
    Nsize = 0
    np.random.seed(seednum)
    d4 = (d1-Nsize)//4
    N1 = np.round(d1*0.8).astype('int')
    N2 = d1 - N1
    X1,y1 = make_moons(n_samples=N1, shuffle=True, noise=0.15, random_state=None)
    X2,y2 = make_moons(n_samples=N2, shuffle=True, noise=0.15, random_state=None)
    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2))
    ind = np.random.permutation(np.arange(d1))
    X = np.array(X[ind,:])                                           
    y = np.array(y[ind])  
    
     # rotaciona   
    tetha = 2.0
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X2 = np.array(X[(Nsize+(d4)*1):(Nsize+(d4)*2),:]) @ Mrot
    
    tetha = 3.95
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X3 = np.array(X[(Nsize+(d4)*2):(Nsize+(d4)*3),:]) @ Mrot
    
    tetha = 5.9
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X4 = np.array(X[(Nsize+(d4)*3):(Nsize+(d4)*4),:]) @ Mrot
    
    #fig, (p1,p2,p3,p4) = plt.subplots(1, 4, figsize=(20, 4))
    #p1.scatter(X[0:(Nsize+(d4)*1),0],X[0:(Nsize+(d4)*1),1], c=y[0:(Nsize+(d4)*1)])
    #p1.axis([-2.5, 2.5, -2.5, 2.5])  
    #p2.scatter(X2[:,0],X2[:,1], c=y[(Nsize+(d4)*1):(Nsize+(d4)*2)])
    #p2.axis([-2.5, 2.5, -2.5, 2.5]) 
    #p3.scatter(X3[:,0],X3[:,1], c=y[(Nsize+(d4)*2):(Nsize+(d4)*3)])
    #p3.axis([-2.5, 2.5, -2.5, 2.5]) 
    #p4.scatter(X4[:,0],X4[:,1], c=y[(Nsize+(d4)*3):(Nsize+(d4)*4)])
    #p4.axis([-2.5, 2.5, -2.5, 2.5])  
    #fig.suptitle('Two Moons', fontsize=25,fontweight='bold')
    #fig.show()
    #plt.show()
    
    X = np.vstack((X[0:(Nsize+(d4)*1),:],X2,X3,X4))
    y[y==0] = -1 
    y = y.astype('int')
    #y = np.hstack((y[0:1100],y[1000:N]))
    y = y.reshape(y.shape[0],1)
    TrueDrift = np.zeros(d1) - 1
    TrueDrift[2500] = 1
    TrueDrift[5000] = 1
    TrueDrift[7500] = 1
    
    return X,y,TrueDrift


###########################################################


def LoadChessboardRot(Nsize,d1): 
    np.random.seed(777)
    #Nsize = 400
    #d1 = 10000 + Nsize
    d4 = (d1-Nsize)//4
    n = 2
    delt1 = np.array([0,1,2,0,1,2,0,1,2])
    delt2 = np.array([0,0,0,1,1,1,2,2,2])
    c1 =np.array([0,1,0,
                  1,0,1,
                  0,1,0,
                  1,0,1])
    X = np.empty([0,n])
    y = np.empty([0])
    for i in np.arange(0,9): 
        x11 = np.random.rand(d1,1) + delt1[i]    
        x12 = np.random.rand(d1,1) + delt2[i]
        X1 = np.concatenate((x11,x12), axis=1)
        y1 = np.repeat(c1[i],d1)
        X = np.vstack((X,X1))  
        y = np.hstack((y,y1))  
    db = np.hstack((X,y.reshape(y.shape[0],1)))
    ind = np.random.permutation(db.shape[0]) 
    ind = np.random.permutation(ind) 
    db = np.array(db[ind,:])  
    
    X = np.array(db[:,0:2])
    y = np.array(db[:,2])
    
    Xmax = np.max(X, axis=0) 
    Xmin = np.min(X, axis=0)  
    X = (X - Xmin)/(Xmax - Xmin + 0.000000001)  -0.5

    X1 = np.array(X[0:(Nsize+(d4)*1),:])
    y1 = np.array(y[0:(Nsize+(d4)*1)])
    X2 = np.array(X[(Nsize+(d4)*1):(Nsize+(d4)*2),:])
    y2 = np.array(y[(Nsize+(d4)*1):(Nsize+(d4)*2)])
    X3 = np.array(X[(Nsize+(d4)*2):(Nsize+(d4)*3),:])
    y3 = np.array(y[(Nsize+(d4)*2):(Nsize+(d4)*3)])
    X4 = np.array(X[(Nsize+(d4)*3):(Nsize+(d4)*4),:])
    y4 = np.array(y[(Nsize+(d4)*3):(Nsize+(d4)*4)])
    
    tetha = 0.8 #0.5
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X2 = np.array(X2[:,0:2]) @ Mrot
    tetha = 1.6 #1
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X3 = np.array(X3[:,0:2]) @ Mrot
    tetha = 2.4 #1.6
    Mrot = np.array([np.cos(tetha),np.sin(tetha),-np.sin(tetha),np.cos(tetha)]).reshape(2,2)
    X4 = np.array(X4[:,0:2]) @ Mrot
    
    #fig, (p1,p2,p3,p4) = plt.subplots(1, 4, figsize=(20, 4))
    #p1.scatter(X1[:,0],X1[:,1], c=y1)
    #p1.axis([-0.75, 0.75, -0.75, 0.75])  
    #p2.scatter(X2[:,0],X2[:,1], c=y2)
    #p2.axis([-0.75, 0.75, -0.75, 0.75]) 
    #p3.scatter(X3[:,0],X3[:,1], c=y3)
    #p3.axis([-0.75, 0.75, -0.75, 0.75]) 
    #p4.scatter(X4[:,0],X4[:,1], c=y4)
    #p4.axis([-0.75, 0.75, -0.75, 0.75]) 
    #fig.suptitle('Checkerboard', fontsize=25,fontweight='bold')
    #fig.show()
    #plt.show()
    
    
    X = np.vstack((X1,X2,X3,X4))  
    y = np.hstack((y1,y2,y3,y4)) 
        
    y = y.astype('int')
    y = y.reshape(y.shape[0],1)
    y[y==0] = -1 
    TrueDrift = np.zeros(X.shape[0]) - 1
    TrueDrift[2500] = 1
    TrueDrift[5000] = 1
    TrueDrift[7500] = 1
    return X,y,TrueDrift