#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 15 08:00:12 2021

@author: https://scikit-multiflow.readthedocs.io/
"""

# Imports
from skmultiflow.data import DataStream
import pandas as pd
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.bayes import NaiveBayes
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import time

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

X, y = Gaussians4dMov(777)
dataset = '4Dgauss4class'


Ncut = 1
#X,y,_ = SpiralsAbruptRot(777,10000)
#X,y,_ = TwoGaussAbruptRot(777,10000)
#X,y,_ = CheckerAbruptRot(777,10000)
#X,y,_ = TwoMoonsAbruptRot(777,10000)
#dataset = 'moons'

#X,y = TwoGaussIncre(777)
#X,y = CheckerRotIncre(777)
#X,y = TwoMoonsRot(777)
#X,y = SpiralsRot(777)
#dataset = "Incspira" 

y[y==-1]=0

N = X.shape[0]
n = X.shape[1]

stream = DataStream(data=X,y=y,n_targets=1)

path = '/home/wagner/mygitlab/Funcs/allfunctions/Tese/DWM/grid/'
grid = pd.read_csv(path + 'gridDWM_' + dataset,header=None)
grid = np.array(grid)

indmax = np.argmax(grid[:,0])
p1 = grid[indmax,1]
p2 = grid[indmax,2]
p3 = grid[indmax,3]
p4 = grid[indmax,4]

print(grid[indmax,:])

starttime = time.time()

dwm = DynamicWeightedMajorityClassifier(n_estimators=p1, 
                                        base_estimator=NaiveBayes(nominal_attributes=None), 
                                        period=p2, 
                                        beta=p3, 
                                        theta=p4)
 
n_samples = 0
correct_cnt = 0
yhat = np.zeros(N)
error = np.zeros(N)

yhat_prob = np.zeros((N,4))
y_dummy = np.zeros((N,4))
 
for i in range(N):

    Xi, yi = stream.next_sample()

    #yaux_prob= dwm.predict_proba(Xi)
    #yhat_prob[i,0] = yaux_prob[1]
    #yhat_prob[i,1] = yaux_prob[2]
    #yhat_prob[i,2] = yaux_prob[3]
    #yhat_prob[i,3] = yaux_prob[4]
    y_dummy[i,yi-1] = 1

    #print (y_dummy[i,:],yhat_prob[i,:])


    yhataux = dwm.predict(Xi)
    yhat[i] = yhataux
    yhat_prob[i,yhataux-1] = 1
    if yi == yhataux:
        correct_cnt = correct_cnt + 1
    else:
        error[i] = 1
    dwm.partial_fit(Xi, yi)
    n_samples = n_samples + 1

Acc = np.array(correct_cnt / n_samples)
#AUC = roc_auc_score(y, yhat)                               # comente para dataset 4D
AUC = roc_auc_score(y_dummy, yhat_prob, multi_class='ovr')
F1 = 0 #f1_score(y_true=y, y_pred=yhat)

endtime = time.time()
Timecount = endtime - starttime

result = np.array([Acc, AUC, F1, Timecount]).reshape(1,4)
#result = np.array([Acc, AUC, F1]).reshape(1,3)
print('Acc - AUC - F1')
print(result)
                
                
path = '/home/wagner/mygitlab/Funcs/allfunctions/Tese/DWM/resultpc/'
np.savetxt(path + 'testDWM_' + dataset, result, delimiter=',' , fmt='%.6f') 
np.savetxt(path + 'testDWM_error_' + dataset, error, delimiter=',' , fmt='%.1d')

