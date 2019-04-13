import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

x_train = np.load('MNIST_train_img.npy')
y_train = np.load('MNIST_train_label.npy')[...,None]

x_test = np.load('MNIST_test_img.npy')
y_test = np.load('MNIST_test_label.npy')[...,None]

#x perturbation
#with tf.device('/gpu:2'):
model = load_model("training_lr1en4_MNIST_ResNet50_epoch151to200_dropout_128_best.h5") 

  
###Setup 2: weights wrt cross entropy
n_layers = len(model.layers)
n_train = int(x_train.shape[0]/1000.0)

FI_train_layers=-np.ones([n_train, n_layers]) # FI for each layer: if a layer has no trainable parameters, return -1.

n_part_param_layers = [len(model.layers[L].trainable_weights) for L in range(n_layers)]# how many parts of parameters that each layer has               
index_layers_n0 = np.nonzero(n_part_param_layers)[0] # indices for layers with trainable parameters

n_layers_n0 = len(index_layers_n0)

trainable_count = np.zeros(n_layers_n0,dtype=int) # the number of trainable parameters that each layer has

for i in range(n_layers_n0):
    layer_id = index_layers_n0[i]
    trainable_count[i] = np.sum([K.count_params(model.layers[ layer_id ].trainable_weights[p]) for p in range(n_part_param_layers[layer_id])])
    
trainable_count_id_start_end = np.insert(np.cumsum(trainable_count),0,0)
  
t = 0
FI_train_all = np.zeros(n_train) # FI for all parameters


grad_K=tf.concat(axis=1,values=[
   tf.concat(axis=0, values=[K.flatten(b)[...,None] for b in K.gradients(model.output[0,k], model.trainable_weights)]) 
   for k in range(n_class)]
   )
            
iterate = K.function([model.input], [grad_K, model.output])  


for i in range(int(n_train*(set_part-1)),int(n_train*set_part)):
     
    grad, pred_P = iterate([x_train[i][None]])
    
    L0 = grad @ np.diag( ((pred_P[0,])**0.5+epsilon)**-1)
    
    f_grad = ( grad[:,y_train[i]]/(pred_P[0,y_train[i]]+epsilon) ).T # shape=(1,...)
    
    #All trainable parameters
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
        
        FI_train_all[t] =  nabla_f @ nabla_f.T

        
    #Each layer's trainable parameters    
    for ell in range(n_layers_n0):    
        id_start, id_end = trainable_count_id_start_end[[ell,ell+1]]
        L0_ell = L0[id_start:id_end,:]
        f_grad_ell = f_grad[:, id_start:id_end]
        
        B0, D_L0, A0 = sp.linalg.svd(L0_ell, full_matrices=False)
        rank_L0 = sum(D_L0>epsilon)
        
        if rank_L0>0:
            B0 = B0[:,:rank_L0]    
            A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0,:]   
            
            U_A, D_0, _ = sp.linalg.svd( A0 @ A0.T, full_matrices=True)
            D_0_inv = np.diag(D_0**-1)
            D_0_inv_sqrt = np.diag(D_0**-0.5)
            
            U_0 = B0 @ U_A
                        
            nabla_f = f_grad_ell @ U_0 @ D_0_inv_sqrt
            
            FI_train_layers[t,index_layers_n0[ell]] =  nabla_f @ nabla_f.T
        else:
            FI_train_layers[t,index_layers_n0[ell]] = 0

            
    t = t + 1

np.savetxt('MNIST_ResNet50_128_best_setup2_part'+ str(set_part) +'_all.txt', FI_train_all, delimiter=' ')
np.savetxt('MNIST_ResNet50_128_best_setup2_part'+ str(set_part) +'_layers.txt', FI_train_layers, delimiter=' ')    
