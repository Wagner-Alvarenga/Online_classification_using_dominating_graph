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
X,y = SpiralsRot(777)   
X2,y2 = SpiralsRot(888)              # Inc  

#X,y = Gaussians4dMov(777)
#X2,y2 = Gaussians4dMov(888)

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
dbname = 'Incspira'
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

    for i in range(max_samples):

        Xi, yi = next(data)
        #print(Xi, yi)
        if n_samples > wait_samples:
            yaux = treemodel.predict_one(Xi)
            yhat[i] = yaux

            if yi == yaux:
                correct_cnt = correct_cnt + 1
            else:
                error[i,j] = 1

        treemodel.learn_one(Xi, yi)

        n_samples = n_samples + 1

    Acc = correct_cnt / (n_samples-wait_samples)
    AUC = roc_auc_score(y, yhat)
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

    for i in range(max_samples):

        Xi, yi = next(data)

        if n_samples > wait_samples:
            yaux = treemodel.predict_one(Xi)
            yhat[i] = yaux

            if yi == yaux:
                correct_cnt = correct_cnt + 1
            else:
                error[i,j] = 1

        treemodel.learn_one(Xi, yi)

        n_samples = n_samples + 1

    Acc = correct_cnt / (n_samples-wait_samples)
    AUC = roc_auc_score(y, yhat)
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

    for i in range(max_samples):

        Xi, yi = next(data)

        if n_samples > wait_samples:
            yaux = treemodel.predict_one(Xi)
            yhat[i] = yaux

            if yi == yaux:
                correct_cnt = correct_cnt + 1
            else:
                error[i,j] = 1

        treemodel.learn_one(Xi, yi)

        n_samples = n_samples + 1

    Acc = correct_cnt / (n_samples-wait_samples)
    AUC = roc_auc_score(y, yhat)
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
