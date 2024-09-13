#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 15 08:00:12 2021

@author: https://scikit-multiflow.readthedocs.io/
"""

# Imports
from river import tree
from river import stream
import pandas as pd
#from old_mydatasets import LoadTwoMoonsRot, LoadTwoRotSpirals, LoadTwoMovGauss, LoadChecker4Rot 
from mydatasets import TwoGaussIncre, CheckerRotIncre, TwoMoonsRot, SpiralsRot 
from mydatasets import SpiralsAbruptRot, CheckerAbruptRot, TwoGaussAbruptRot, TwoMoonsAbruptRot
import numpy as np
from sklearn.metrics import roc_auc_score
import time
from mydatasets2 import Gaussians4dMov


#--------------------------------------------------------------------------

#Ncut = 1

##X = pd.read_csv("/home/wagner/datasets/Checkerboard_X.csv",header=None)
##y = pd.read_csv("/home/wagner/datasets/Checkerboard_y.csv",header=None)
#X = np.array(X)
#y = np.array(y)
#y[y==-1]=0
## num_s_test = 1000
## X = np.array(X[0:num_s_test,:])
## y = np.array(y[0:num_s_test,:])
#y = y.reshape(y.shape[0]).astype('int')

################

#X,y,TrueDrift = SpiralsAbruptRot(777,10000)       # 10.000
#X2,y2,TrueDrift = SpiralsAbruptRot(888,10000) 
#X,y,TrueDrift = CheckerAbruptRot(777,10000)       # 10.000
#X2,y2,TrueDrift = CheckerAbruptRot(888,10000)   
#X,y,TrueDrift = TwoGaussAbruptRot(777,10000)      # 10.000
#X2,y2,TrueDrift = TwoGaussAbruptRot(888,10000) 
#X,y,TrueDrift = TwoMoonsAbruptRot(777,10000)
#X2,y2,TrueDrift = TwoMoonsAbruptRot(888,10000)

#X,y = TwoGaussIncre(777)             # Inc           
#X2,y2 = TwoGaussIncre(888)  
#X,y = CheckerRotIncre(777)   
#X2,y2 = CheckerRotIncre(888)         # Inc   
#X,y = TwoMoonsRot(777) 
#X2,y2 = TwoMoonsRot(888) 
#X,y = SpiralsRot(777)   
#X2,y2 = SpiralsRot(888)              # Inc  

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

X,y = Gaussians4dMov(777)
X2,y2 = Gaussians4dMov(888)

wait300 = 300
X2 = X2[0:wait300,:]
y2 = y2[0:wait300,:]

X = np.vstack((X2,X))
y = np.vstack((y2,y)) 

print(np.unique(y,return_counts=True))
y[y==0]=-1
y = y.reshape(y.shape[0]).astype('int')
#Wsize = 200
#xtrain = np.array(xtrain[0:Wsize,:])
#ytrain = np.array(ytrain[0:Wsize,:])


def Streamdata(X,y):
    data = stream.iter_array(X,y,shuffle=False)
    return data

#--------------------------------------------------------------------------

#dbname = '4Dgauss_'
dbname = '4Dgauss4class'
#dbname = 'spira'
#dbname = '4Dgauus'

dbname2 = dbname
path = '/home/wagner/mygitlab/Funcs/allfunctions/Tese/HoeffdingTree/grid/'

gridgain = pd.read_csv(path + 'gridGainAdapte_' + dbname2,header=None)
gridgini = pd.read_csv(path + 'gridGiniAdapte_' + dbname2,header=None)
gridhell = pd.read_csv(path + 'gridHellAdapte_' + dbname2,header=None)
gridgain = np.array(gridgain)
gridgini = np.array(gridgini)
gridhell = np.array(gridhell)

indmaxgain = np.argmax(gridgain[:,0])
p1gain = gridgain[indmaxgain,1]
p2gain = gridgain[indmaxgain,2]
p3gain = gridgain[indmaxgain,3]
p4gain = gridgain[indmaxgain,4]

indmaxgini = np.argmax(gridgini[:,0])
p1gini = gridgini[indmaxgini,1]
p2gini = gridgini[indmaxgini,2]
p3gini = gridgini[indmaxgini,3]
p4gini = gridgini[indmaxgini,4]

indmaxhell = np.argmax(gridhell[:,0])
p1hell = gridhell[indmaxhell,1]
p2hell = gridhell[indmaxhell,2]
p3hell = gridhell[indmaxhell,3]
p4hell = gridhell[indmaxhell,4]

print('grid gain',gridgain[indmaxgain,:])
print('grid gini',gridgini[indmaxgini,:])
print('grid hellinher',gridhell[indmaxhell,:])

#--------------------------------------------------------------------------

repete = 1

#--------------------------------------------------------------------------

result = np.zeros((repete,4))
error  = np.zeros((X.shape[0],repete))

for j in range(repete):

    starttime = time.time()
    
    data = Streamdata(X,y)

    treemodel = tree.HoeffdingAdaptiveTreeClassifier(
                                        split_criterion = 'info_gain',
                                        tie_threshold = p4gain,
                                        leaf_prediction = 'mc',
                                        bootstrap_sampling = False,
                                        binary_split = False,
                                        seed = j,
                                        grace_period = p1gain, 
                                        max_depth = p2gain,
                                        split_confidence = p3gain, 
                                        drift_window_threshold = 300, 
                                        adwin_confidence = 0.002
                                        )
    
    n_samples = 0
    correct_cnt = 0    
    wait_samples = wait300
    max_samples = X.shape[0]
    
    yhat = np.zeros(max_samples)  
    yhat_prob = np.zeros((max_samples-wait_samples,4))
    y_dummy = np.zeros((max_samples-wait_samples,4)) 

    for i in range(max_samples):

        Xi, yi = next(data)
        #print(Xi, yi)
        if n_samples > wait_samples:
            yaux = treemodel.predict_one(Xi)
            
            yaux_prob = treemodel.predict_proba_one(Xi)
            yhat_prob[wait_samples-i,0] = yaux_prob[1]
            yhat_prob[wait_samples-i,1] = yaux_prob[2]
            yhat_prob[wait_samples-i,2] = yaux_prob[3]
            yhat_prob[wait_samples-i,3] = yaux_prob[4]
            y_dummy[wait_samples-i,yi-1] = 1

            yhat[i] = yaux

            if yi == yaux:
                correct_cnt = correct_cnt + 1
            else:
                error[i,j] = 1

        treemodel.learn_one(Xi, yi)

        n_samples = n_samples + 1

    Acc = correct_cnt / (n_samples-wait_samples)
    AUC = roc_auc_score(y_dummy, yhat_prob, multi_class='ovr')
    #F1 = f1_score(y_true=y, y_pred=yhat)

    endtime = time.time()
    Timecount = endtime - starttime
    
    result[j,0] = Acc
    result[j,1] = AUC
    #result[j,2] = F1
    result[j,3] = Timecount

error = error[wait300:]

print('Gain:  Acc - AUC - F1 - Time')
print(np.mean(result,axis=0))
                
                
path = '/home/wagner/mygitlab/Funcs/allfunctions/Tese/HoeffdingTree/resultpc/'
np.savetxt(path + 'testGAIN_' + dbname, result, delimiter=',' , fmt='%.6f') 
np.savetxt(path + 'testGAIN_error_' + dbname, error, delimiter=',' , fmt='%.1d')

#--------------------------------------------------------------------------

result = np.zeros((repete,4))
error  = np.zeros((X.shape[0],repete))

for j in range(repete):

    starttime = time.time()
    
    data = Streamdata(X,y)

    treemodel = tree.HoeffdingAdaptiveTreeClassifier(
                                        split_criterion = 'gini',
                                        tie_threshold = p4gini,
                                        leaf_prediction = 'mc',
                                        bootstrap_sampling = False,
                                        binary_split = False,
                                        seed = j,
                                        grace_period = p1gini, 
                                        max_depth = p2gini,
                                        split_confidence = p3gini, 
                                        drift_window_threshold = 300, 
                                        adwin_confidence = 0.002
                                        )
    
    n_samples = 0
    correct_cnt = 0    
    wait_samples = wait300
    max_samples = X.shape[0]
    
    yhat = np.zeros(max_samples)  
    yhat_prob = np.zeros((max_samples-wait_samples,4))
    y_dummy = np.zeros((max_samples-wait_samples,4)) 


    for i in range(max_samples):

        Xi, yi = next(data)

        if n_samples > wait_samples:
            yaux = treemodel.predict_one(Xi)

            yaux_prob = treemodel.predict_proba_one(Xi)
            yhat_prob[wait_samples-i,0] = yaux_prob[1]
            yhat_prob[wait_samples-i,1] = yaux_prob[2]
            yhat_prob[wait_samples-i,2] = yaux_prob[3]
            yhat_prob[wait_samples-i,3] = yaux_prob[4]
            y_dummy[wait_samples-i,yi-1] = 1


            yhat[i] = yaux

            if yi == yaux:
                correct_cnt = correct_cnt + 1
            else:
                error[i,j] = 1

        treemodel.learn_one(Xi, yi)

        n_samples = n_samples + 1

    Acc = correct_cnt / (n_samples-wait_samples)
    AUC = roc_auc_score(y_dummy, yhat_prob, multi_class='ovr')
    #F1 = f1_score(y_true=y, y_pred=yhat)

    endtime = time.time()
    Timecount = endtime - starttime
    
    result[j,0] = Acc
    result[j,1] = AUC
    #result[j,2] = F1
    result[j,3] = Timecount

error = error[wait300:]

print('Gini:  Acc - AUC - F1 - Time')
print(np.mean(result,axis=0))
                
                

np.savetxt(path + 'testGINI_' + dbname, result, delimiter=',' , fmt='%.6f') 
np.savetxt(path + 'testGINI_error_' + dbname, error, delimiter=',' , fmt='%.1d')

#--------------------------------------------------------------------------

result = np.zeros((repete,4))
error  = np.zeros((X.shape[0],repete))

for j in range(repete):

    starttime = time.time()
    
    data = Streamdata(X,y)

    treemodel = tree.HoeffdingAdaptiveTreeClassifier(
                                        split_criterion = 'hellinger',
                                        tie_threshold = p4hell,
                                        leaf_prediction = 'mc',
                                        bootstrap_sampling = False,
                                        binary_split = False,
                                        seed = j,
                                        grace_period = p1hell, 
                                        max_depth = p2hell,
                                        split_confidence = p3hell, 
                                        drift_window_threshold = 300, 
                                        adwin_confidence = 0.002
                                        )
    
    n_samples = 0
    correct_cnt = 0    
    wait_samples = wait300
    max_samples = X.shape[0]
    
    yhat = np.zeros(max_samples)  
    yhat_prob = np.zeros((max_samples-wait_samples,4))
    y_dummy = np.zeros((max_samples-wait_samples,4)) 

    for i in range(max_samples):

        Xi, yi = next(data)

        if n_samples > wait_samples:
            yaux = treemodel.predict_one(Xi)

            yaux_prob = treemodel.predict_proba_one(Xi)
            yhat_prob[wait_samples-i,0] = yaux_prob[1]
            yhat_prob[wait_samples-i,1] = yaux_prob[2]
            yhat_prob[wait_samples-i,2] = yaux_prob[3]
            yhat_prob[wait_samples-i,3] = yaux_prob[4]
            y_dummy[wait_samples-i,yi-1] = 1


            yhat[i] = yaux

            if yi == yaux:
                correct_cnt = correct_cnt + 1
            else:
                error[i,j] = 1

        treemodel.learn_one(Xi, yi)

        n_samples = n_samples + 1

    Acc = correct_cnt / (n_samples-wait_samples)
    AUC = roc_auc_score(y_dummy, yhat_prob, multi_class='ovr')
    #F1 = f1_score(y_true=y, y_pred=yhat)

    endtime = time.time()
    Timecount = endtime - starttime
    
    result[j,0] = Acc
    result[j,1] = AUC
    #result[j,2] = F1
    result[j,3] = Timecount

error = error[wait300:]

print('Hellinger:  Acc - AUC - F1 - Time')
print(np.mean(result,axis=0))
                
                

np.savetxt(path + 'testHELL_' + dbname, result, delimiter=',' , fmt='%.6f') 
np.savetxt(path + 'testHELL_error_' + dbname, error, delimiter=',' , fmt='%.1d')

#--------------------------------------------------------------------------
