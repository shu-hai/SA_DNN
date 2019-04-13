import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
set_part = int(sys.argv[1])

import numpy as np
import scipy as sp
import tensorflow as tf



import keras
from keras.models import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint,CSVLogger, Callback
from keras.utils.training_utils import multi_gpu_model

from keras import backend as K


np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import random
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

epsilon = 1e-3

n_class = 10

K.set_learning_phase(0)


#import time

x_train = np.load('CIFAR10_train_img.npy')
y_train = np.load('CIFAR10_train_label.npy')
y_train_pred = np.load('CIFAR10_ResNet50_train_pred_label.npy')# predicted label

x_test = np.load('CIFAR10_test_img.npy')
y_test_pred = np.load('CIFAR10_ResNet50_test_pred_label.npy')

#x perturbation
#with tf.device('/cpu:0'):
model = load_model("training_lr1en4_CIFAR10_ResNet50_epoch401to500_dropout_128_best.h5") 
    
    
###Setup 3: training and test sets wrt sum squared prob


grad_K=tf.concat(axis=1,values=[
   tf.concat(axis=0, values=[K.flatten(b)[...,None] for b in K.gradients(model.output[0,k], model.input)]) 
   for k in range(n_class)]
   )
            
iterate = K.function([model.input], [grad_K, model.output])  

              
FI_train=np.zeros(int(x_train.shape[0]/1000.0))
FI_train_pred=np.zeros(int(x_train.shape[0]/1000.0))

#for i in range(x_train.shape[0]):
t = 0    
for i in range(int(x_train.shape[0]/1000.0*(set_part-1)),int(x_train.shape[0]/1000.0*set_part)):
          
    grad, pred_P = iterate([x_train[i][None]])
    
    L0 = grad @ np.diag( ((pred_P[0,])**0.5+epsilon)**-1)
    
    f_grad = ( grad[:,y_train[i]]/(pred_P[0,y_train[i]]+epsilon) ).T # shape=(1,...)    
    f_grad_pred = ( grad[:,y_train_pred[i]]/(pred_P[0,y_train_pred[i]]+epsilon) ).T
        
    B0, D_L0, A0 = sp.linalg.svd(L0, full_matrices=False)
    rank_L0 = sum(D_L0>epsilon)
    
    if rank_L0>0:
        B0 = B0[:,:rank_L0]    
        A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0,:]   
        
        U_A, D_0, _ = sp.linalg.svd( A0 @ A0.T, full_matrices=True)
        D_0_inv = np.diag(D_0**-1)
        D_0_inv_sqrt = np.diag(D_0**-0.5)
        
        U_0 = B0 @ U_A
                    
        nabla_f = f_grad @ U_0 @ D_0_inv_sqrt        
        nabla_f_pred = f_grad_pred @ U_0 @ D_0_inv_sqrt
        
        FI_train[t] =  nabla_f @ nabla_f.T        
        FI_train_pred[t] =  nabla_f_pred @ nabla_f_pred.T
      
        
    t = t + 1



#
FI_test_pred=np.zeros(int(x_test.shape[0]/1000.0))
#for i in range(x_test.shape[0]):
t = 0    
for i in range(int(x_test.shape[0]/1000.0*(set_part-1)),int(x_test.shape[0]/1000.0*set_part)):
    
    grad, pred_P = iterate([x_test[i][None]])
    
    L0 = grad @ np.diag( ((pred_P[0,])**0.5+epsilon)**-1)
    
    f_grad_pred = ( grad[:,y_test_pred[i]]/(pred_P[0,y_test_pred[i]]+epsilon) ).T
        
    B0, D_L0, A0 = sp.linalg.svd(L0, full_matrices=False)
    rank_L0 = sum(D_L0>epsilon)
    
    if rank_L0>0:
        B0 = B0[:,:rank_L0]    
        A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0,:]   
        
        U_A, D_0, _ = sp.linalg.svd( A0 @ A0.T, full_matrices=True)
        D_0_inv = np.diag(D_0**-1)
        D_0_inv_sqrt = np.diag(D_0**-0.5)
        
        U_0 = B0 @ U_A
                          
        nabla_f_pred = f_grad_pred @ U_0 @ D_0_inv_sqrt
              
        FI_test_pred[t] =  nabla_f_pred @ nabla_f_pred.T
      
        
    t = t + 1

       

np.savetxt('CIFAR10_ResNet50_128_best_setup1_part'+ str(set_part) + '_train.txt', FI_train, delimiter=' ')
np.savetxt('CIFAR10_ResNet50_128_best_setup3_part'+ str(set_part) + '_train.txt', FI_train_pred, delimiter=' ')
np.savetxt('CIFAR10_ResNet50_128_best_setup3_part'+ str(set_part) + '_test.txt', FI_test_pred, delimiter=' ')




