#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 15 08:00:12 2021

@author: https://scikit-multiflow.readthedocs.io/
"""

# Imports
from skmultiflow.data import DataStream
import pandas as pd
#from mydatasets import LoadTwoMoonsRot, LoadTwoRotSpirals, LoadTwoMovGauss, LoadChecker4Rot 
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.bayes import NaiveBayes
import numpy as np
from mydatasets import SpiralsAbruptRot, CheckerAbruptRot, TwoGaussAbruptRot, TwoMoonsAbruptRot 
from mydatasets import TwoGaussIncre, CheckerRotIncre, TwoMoonsRot, SpiralsRot


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
    
    ttt=1
    
    c11 = np.array([-6.  ,  0. , -16.5 ,-22.5])   * ttt
    c12 = np.array([-16.5 ,-21. ,  -7.5 , -7.5]) * ttt
    
    c21 = np.array([-3. , -16.5 ,-19.5 , -9.])  * ttt
    c22 = np.array([-25.5 , -6.  , -6. , -16.5])   * ttt
    
    c31 = np.array([-7.5 ,-10.5  , 1.5, -10.5])  * ttt
    c32 = np.array([-13.5 ,-15. ,  -3. , -15. ])   * ttt
    
    c41 = np.array([1.5 , -7.5 ,-22.5 ,-25.5])   * ttt
    c42 = np.array([-27.,  -24. , -10.5  , 3.])   *ttt
    
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
    
    
    # 1>2   3>4
    
    delt11 = delt11/(N/2)
    delt12 = delt12/(N/2)
    delt21 = delt21/(N/2)
    delt22 = delt22/(N/2)
    
    X = np.empty([0,n])
    y = np.empty([0,1])
    
    # X1 = np.empty([0,n])
    # X2 = np.empty([0,n])
    # X3 = np.empty([0,n])
    # X4 = np.empty([0,n])
    
    # y1 = np.empty([0,1])
    # y2 = np.empty([0,1])
    # y3 = np.empty([0,1])
    # y4 = np.empty([0,1])
    
    
    for i in range(N):
        
        x11 = np.array(np.random.randn(1,n) + c11).reshape(1,n)
        x12 = np.array(np.random.randn(1,n) + c12).reshape(1,n)
        x21 = np.array(np.random.randn(1,n) + c21).reshape(1,n)
        x22 = np.array(np.random.randn(1,n) + c22).reshape(1,n)   
        x31 = np.array(np.random.randn(1,n) + c31).reshape(1,n)
        x32 = np.array(np.random.randn(1,n) + c32).reshape(1,n)
        x41 = np.array(np.random.randn(1,n) + c41).reshape(1,n)
        x42 = np.array(np.random.randn(1,n) + c42).reshape(1,n)
        
        # X1 = np.vstack((X1,x11))
        # X1 = np.vstack((X1,x12))
        X1 = np.vstack((x11,x12))
        # y1 = np.vstack((y1,np.repeat(1,c).reshape(c,1)))
        y1 = np.array([1,1]).reshape(2,1)
        
        # X2 = np.vstack((X2,x21))
        # X2 = np.vstack((X2,x22))
        X2 = np.vstack((x21,x22))
        # y2 = np.vstack((y2,np.repeat(2,c).reshape(c,1)))
        y2 = np.array([2,2]).reshape(2,1)
        
        # X3 = np.vstack((X3,x31))
        # X3 = np.vstack((X3,x32))
        X3 = np.vstack((x31,x32))
        # y3 = np.vstack((y3,np.repeat(3,c).reshape(c,1)))
        y3 = np.array([3,3]).reshape(2,1)
        
        # X4 = np.vstack((X4,x41))
        # X4 = np.vstack((X4,x42))
        X4 = np.vstack((x41,x42))
        # y4 = np.vstack((y4,np.repeat(4,c).reshape(c,1)))
        y4 = np.array([4,4]).reshape(2,1)
        
        Xaux = np.vstack((X1,X2,X3,X4))
        yaux = np.vstack((y1,y2,y3,y4))
        
        randind = np.array([0,1,2,3,4,5,6,7])
        np.random.shuffle(randind)
        
        Xaux = Xaux[randind,:]
        yaux = yaux[randind,:]
        X = np.vstack((X,Xaux))
        y = np.vstack((y,yaux))
                
        # X = np.vstack((X,X1,X2,X3,X4))
        # y = np.vstack((y,y1,y2,y3,y4))
        
        if i < N//2:
    
            c31 = c31 - delt11
            c32 = c32 - delt12
            c41 = c41 - delt11*(-1)
            c42 = c42 - delt12*(-1)
            
        elif i == N//2:
            
            c11aux = c11  
            c32aux = c32  
            
            c11 = c21  
            c32 = c42  
            c21 = c11aux  
            c42 = c32aux  
            
        else:                  # arrumar depois
            
            c11 = c11 - delt21
            c12 = c12 - delt22
            c21 = c21 - delt21*(-1)
            c22 = c22 - delt22*(-1)
            
    Xmin = np.min(X,axis=0)
    Xmax = np.max(X,axis=0)                  
    X = (X-Xmin)/(Xmax-Xmin)
    
    # plt.scatter(X[:,0],X[:,1],c=y)
    # plt.show()
    y = y.astype('int')
    y = y.reshape(y.shape[0],1)
    print('Tamanho dos dados:   ',X.shape[0])
    print(np.unique(y,return_counts=True))
    return X, y   

X, y = Gaussians4dMov(888)
dataset = '4Dgauss4class'


Ncut = 1
#X,y,_ = SpiralsAbruptRot(888,20000)
#X,y,_ = TwoGaussAbruptRot(888,20000)
#X,y,_ = CheckerAbruptRot(888,20000)
#X,y,_ = TwoMoonsAbruptRot(888,20000)
#dataset = 'moons'

#X,y = TwoGaussIncre(888)
#X,y = CheckerRotIncre(888)
#X,y = TwoMoonsRot(888)
#X,y = SpiralsRot(888)
#dataset = "Incspira" 

#X = pd.read_csv("/home/wagner/datasets/Checkerboard_X.csv",header=None)
#y = pd.read_csv("/home/wagner/datasets/Checkerboard_y.csv",header=None)
X = np.array(X)
y = np.array(y)
########y[y==-1]=0       #   !!!!!   comentado para 4Dgauss4class  !!!!
num_s_test = 1000
X = X[0:num_s_test,:]
y = y[0:num_s_test,:]

def Streamdata(X,y):
    stream = DataStream(data=X,y=y,n_targets=1)
    return stream

p1 = np.array([5, 10, 15, 20, 25])
p2 = np.array([30, 40, 50, 60, 70])
p3 = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
p4 = np.array([0.005, 0.01, 0.05, 0.1, 0.5])

numtest = 625
Gridresult = np.zeros((numtest,5))   # pn + 1
count = 0

for j1 in range(p1.shape[0]):
    for j2 in range(p2.shape[0]):
        for j3 in range(p3.shape[0]):
            for j4 in range(p4.shape[0]):

                stream = Streamdata(X,y)

                dwm = DynamicWeightedMajorityClassifier(n_estimators=p1[j1], 
                                                        base_estimator=NaiveBayes(nominal_attributes=None), 
                                                        period=p2[j2], 
                                                        beta=p3[j3], 
                                                        theta=p4[j4])
                 
                n_samples = 0
                correct_cnt = 0
                max_samples = num_s_test
                
        
                for i in range(max_samples):
                
                    Xi, yi = stream.next_sample()
                    yhat = dwm.predict(Xi)
                    if yi == yhat:
                        correct_cnt = correct_cnt + 1
                    dwm.partial_fit(Xi, yi)
                    n_samples = n_samples + 1
                
                Acc = np.array(correct_cnt / n_samples)
                
                Gridresult[count,0] = Acc
                Gridresult[count,1] = p1[j1]
                Gridresult[count,2] = p2[j2]
                Gridresult[count,3] = p3[j3]
                Gridresult[count,4] = p4[j4]
                
                print(count, numtest, Gridresult[count,:])
                
                count = count + 1
                
                
path = '/home/wagner/mygitlab/Funcs/allfunctions/Tese/DWM/grid/'
np.savetxt(path + 'gridDWM_' + dataset, Gridresult, delimiter=',' , fmt='%.4f') 


