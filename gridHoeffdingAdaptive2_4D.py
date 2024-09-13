
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
#X,y = SpiralsRot(888)

#X,y = Gaussians4dMov(888)
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

