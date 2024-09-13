
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:19:28 2021

@author:  https://riverml.xyz/
"""

# Imports
import numpy as np
from river import tree
from river import stream
from mydatasets import SpiralsAbruptRot, CheckerAbruptRot, TwoGaussAbruptRot, TwoMoonsAbruptRot 
from mydatasets import TwoGaussIncre, CheckerRotIncre, TwoMoonsRot, SpiralsRot
from mydatasets2 import Gaussians4dMov

Ncut = 1
#X,y,_ = SpiralsAbruptRot(888,20000)
#X,y,_ = TwoGaussAbruptRot(888,20000)
#X,y,_ = CheckerAbruptRot(888,20000)
#X,y,_ = TwoMoonsAbruptRot(888,20000)

#X,y = TwoGaussIncre(888)
#X,y = CheckerRotIncre(888)
#X,y = TwoMoonsRot(888)
X,y = SpiralsRot(888)

#X,y = Gaussians4dMov(888)


dataset = 'Incspira'

#X = pd.read_csv("/home/wagner/datasets/Checkerboard_X.csv",header=None)
#y = pd.read_csv("/home/wagner/datasets/Checkerboard_y.csv",header=None)
X = np.array(X)
y = np.array(y)
y[y==-1]=0
num_s_test = 1000
X = np.array(X[0:num_s_test,:])
y = np.array(y[0:num_s_test,:]).astype('int')
y = y.reshape(y.shape[0])

def Streamdata(X,y):
    data = stream.iter_array(X,y,shuffle=False)
    return data

# tree
#grace_period (int) – defaults to 200 -Number of instances a leaf should observe between split attempts.
#max_depth (int) – defaults to None_ -The maximum depth a tree can reach. If None_, the tree will grow indefinitely.
#split_confidence (float) – defaults to 1e-07 - Allowed error in split decision, a value closer to 0 takes longer to decide.
#tie_threshold (float) – defaults to 0.05 -Threshold below which a split will be forced to break ties.

# adwin
#drift_window_threshold (int) – defaults to 300 - Minimum number of examples an alternate tree must observe before being considered as a pote$
#adwin_confidence (float) – defaults to 0.002 - The delta parameter used in the nodes' ADWIN drift detectors.

p1 = [25,50,75,100,125,150,175,200,225,250]                 # 10
p2 = [4,8,10,12,14,16,20,22,24,26]                          # 10 
p3 = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1]     # 7
p4 = [0.0005,0.0001,0.005,0.001,0.05,0.01,0.5]              # 7

#p5 = [150,300,450] 
#p6 = [0.0002,0.002,0.02] 

numtest = 4900   #  81 - 256 - 625
Gridgini = np.zeros((numtest,6)) 
Gridgain = np.zeros((numtest,6))  
Gridhell = np.zeros((numtest,6))  
count = 0

for j1 in range(len(p1)):
    for j2 in range(len(p2)):
        for j3 in range(len(p3)):
            for j4 in range(len(p4)):

                data = Streamdata(X,y)

                ginitree = tree.HoeffdingAdaptiveTreeClassifier(
                                                        split_criterion = 'gini',
                                                        tie_threshold = p4[j4],
                                                        leaf_prediction = 'mc',
                                                        bootstrap_sampling = False,
                                                        binary_split = False,
                                                        seed = 777,
                                                        grace_period = p1[j1], 
                                                        max_depth = p2[j2],
                                                        split_confidence = p3[j3], 
                                                        #drift_window_threshold = p5[j5] 
                                                        #adwin_confidence = p5[j5]
                                                        )
                gaintree = tree.HoeffdingAdaptiveTreeClassifier( 
                                                        split_criterion = 'info_gain',
                                                        tie_threshold = p4[j4],
                                                        leaf_prediction = 'mc',
                                                        bootstrap_sampling = False,
                                                        binary_split = False,
                                                        seed = 777,
                                                        grace_period = p1[j1], 
                                                        max_depth = p2[j2],
                                                        split_confidence = p3[j3], 
                                                        #drift_window_threshold = p5[j5] 
                                                        #adwin_confidence = p5[j5]
                                                        )
                    
                helltree = tree.HoeffdingAdaptiveTreeClassifier( 
                                                        split_criterion = 'hellinger',
                                                        tie_threshold = p4[j4],
                                                        leaf_prediction = 'mc',
                                                        bootstrap_sampling = False,
                                                        binary_split = False,
                                                        seed = 777,
                                                        grace_period = p1[j1], 
                                                        max_depth = p2[j2],
                                                        split_confidence = p3[j3], 
                                                        #drift_window_threshold = p5[j5] 
                                                        #adwin_confidence = p5[j5]
                                                        )

                n_samples = 0
                correct_cnt1 = 0    # gini
                correct_cnt2 = 0    # gain
                correct_cnt3 = 0    # hellinger
                max_samples = 1000
                wait_samples = 300

                yhat1 = np.zeros(max_samples) + 3
                yhat2 = np.zeros(max_samples) + 3
                yhat3 = np.zeros(max_samples) + 3

                for i in range(max_samples):
    
                    Xi, yi = next(data)
                    #print(Xi,yi)
                    if n_samples > wait_samples:
                        yaux1 = ginitree.predict_one(Xi)
                        yaux2 = gaintree.predict_one(Xi)
                        yaux3 = helltree.predict_one(Xi)
                        yhat1[i] = yaux1
                        yhat2[i] = yaux2
                        yhat3[i] = yaux3
                        #print(yi,yhat1[i],yhat2[i],correct_cnt1,correct_cnt2)
                        if yi == yaux1:
                            correct_cnt1 = correct_cnt1 + 1
                        if yi == yaux2:
                            correct_cnt2 = correct_cnt2 + 1
                        if yi == yaux3:
                            correct_cnt3 = correct_cnt3 + 1
                    ginitree.learn_one(Xi, yi)
                    gaintree.learn_one(Xi, yi)
                    helltree.learn_one(Xi, yi)
                    n_samples = n_samples + 1

                Acc1 = correct_cnt1 / (n_samples-wait_samples)
                Acc2 = correct_cnt2 / (n_samples-wait_samples)
                Acc3 = correct_cnt3 / (n_samples-wait_samples)


                Gridgini[count,0] = Acc1
                Gridgini[count,1] = p1[j1]
                Gridgini[count,2] = p2[j2]
                Gridgini[count,3] = p3[j3]
                Gridgini[count,4] = p4[j4]
                #Gridgini[count,5] = p5[j5]

                Gridgain[count,0] = Acc2
                Gridgain[count,1] = p1[j1]
                Gridgain[count,2] = p2[j2]
                Gridgain[count,3] = p3[j3]
                Gridgain[count,4] = p4[j4]
                #Gridgain[count,5] = p5[j5]


                Gridhell[count,0] = Acc3
                Gridhell[count,1] = p1[j1]
                Gridhell[count,2] = p2[j2]
                Gridhell[count,3] = p3[j3]
                Gridhell[count,4] = p4[j4]
                #Gridhell[count,5] = p5[j5]
               
                #print('Gini',  Gridgini[np.argmax(Gridgini[0,:]),:])
                #print('Gain',  Gridgain[np.argmax(Gridgain[0,:]),:])
                #print('Hellinger', Gridhell[np.argmax(Gridhell[0,:]),:])


                #print('Gini', count, numtest, Gridgini[count,:])
                #print('Gain', count, numtest, Gridgain[count,:])
                #print('Hellinger', count, numtest, Gridhell[count,:])
                 
                count = count + 1


path = '/home/wagner/mygitlab/Funcs/allfunctions/Tese/HoeffdingTree/grid/'
np.savetxt(path + 'gridGiniAdapte_' + dataset, Gridgini, delimiter=',' , fmt='%.9f')   # checar casas decimais
np.savetxt(path + 'gridGainAdapte_' + dataset, Gridgain, delimiter=',' , fmt='%.9f')
np.savetxt(path + 'gridHellAdapte_' + dataset, Gridhell, delimiter=',' , fmt='%.9f')


print('Gini',  Gridgini[np.argmax(Gridgini[:,0]),:])
print('Gain',  Gridgain[np.argmax(Gridgain[:,0]),:])
print('Hellinger', Gridhell[np.argmax(Gridhell[:,0]),:])

