#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:52:26 2018

@author: hshu
"""

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
#set_part = int(sys.argv[1])

import numpy as np
import scipy as sp
import tensorflow as tf



import keras#2.1.5
from keras.models import *

from keras import backend as K


np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import random
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

epsilon = 1e-8

n_class = 10

K.set_learning_phase(0)


#import time

x_outlier = np.load('MNIST_train_img_Outlier.npy')
y_outlier = np.load('MNIST_train_label_Outlier.npy')[...,None]
num_outlier = y_outlier.shape[0]

x_n_outlier = np.load('MNIST_train_img.npy')
y_n_outlier = np.load('MNIST_train_label.npy')[...,None]
num_n_outlier = y_n_outlier.shape[0]

#x perturbation

model = load_model("training_lr1en4_MNIST_DenseNet121_epoch151to200_dropout_128_best.h5") 
grad_K=tf.concat(axis=1,values=[
           K.flatten(K.gradients(model.output[0,k], model.input)[0][0])[...,None]
   for k in range(n_class)]
   )#784*10

#img_size = model.input.shape[1]*model.input.shape[2]

#grad2_K=tf.concat(axis=2,values=[
#   tf.concat(axis=1, values=[           
#           K.flatten(K.gradients(grad_K[i,k], model.input)[0][0])[...,None]
#           for i in range(img_size)])[...,None]
#           for k in range(n_class)])


iterate = K.function([model.input], [grad_K, model.output])  
   
##non-outlier       
FI_n_outlier = np.zeros(num_n_outlier)
Jacob_n_outlier = np.zeros(num_n_outlier)#Jacobian norm
#C_n_outlier = np.zeros(40)#max Cook's local influence

FI_outlier = np.zeros(num_outlier)
Jacob_outlier = np.zeros(num_outlier)#Jacobian norm
#C_outlier = np.zeros(40)#max Cook's local influence

#for i in range(x_train.shape[0]):
#t = 0
#for i in range(int(40*(set_part-1)),int(40*set_part)):
for i in range(num_n_outlier):         
    
    #Non-Outlier
    grad, pred_P = iterate([x_n_outlier[i][None]])
    
    L0 = grad @ np.diag( ((pred_P[0,])**0.5+epsilon)**-1)
    
    f_grad = ( grad[:,y_n_outlier[i]]/(pred_P[0,y_n_outlier[i]]+epsilon) ).T # shape=(1,...)    
    
    
    
    #Jacobian norm
    Jacob_n_outlier[i] = np.linalg.norm(f_grad,'fro')
    
    #max Cook's local influence    
    #flatten(x)=tf.reshape(x, [-1])=np.reshape(x, [-1]); see https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
    '''    
    H_f = - f_grad.T @ f_grad + grad2[:,:,y_n_outlier[i]] / (pred_P[0,y_n_outlier[i]]+epsilon)
    
    G_f = np.eye(f_grad.shape[1])+ f_grad.T @ f_grad
    
    U_G, D_G, _ = sp.linalg.svd( G_f, full_matrices=True)
    
    D_G_inv_sqrt = np.diag(D_G**-0.5)
    
    G_f_inv_sqrt = U_G @ D_G_inv_sqrt @ U_G.T
    
    C_n_outlier[i] = sp.linalg.svdvals(G_f_inv_sqrt @ H_f @ G_f_inv_sqrt)[0] / np.sqrt(1 + f_grad @ f_grad.T)
    '''
    
    #FI
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
        
        FI_n_outlier[i] =  nabla_f @ nabla_f.T        

    i = i + 1
    print(i)
        
        
        
    #Outlier    
for i in range(num_outlier):          
    grad, pred_P = iterate([x_outlier[i][None]])
    
    L0 = grad @ np.diag( ((pred_P[0,])**0.5+epsilon)**-1)
    
    f_grad = ( grad[:,y_outlier[i]]/(pred_P[0,y_outlier[i]]+epsilon) ).T # shape=(1,...)    
    
    
    
    #Jacobian norm
    Jacob_outlier[i] = np.linalg.norm(f_grad,'fro')
    
    #max Cook's local influence    
    #flatten(x)=tf.reshape(x, [-1])=np.reshape(x, [-1]); see https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
    '''
    H_f = - f_grad.T @ f_grad + grad2[:,:,y_outlier[i]] / (pred_P[0,y_outlier[i]]+epsilon)
    
    G_f = np.eye(f_grad.shape[1])+ f_grad.T @ f_grad
    
    U_G, D_G, _ = sp.linalg.svd( G_f, full_matrices=True)
    
    D_G_inv_sqrt = np.diag(D_G**-0.5)
    
    G_f_inv_sqrt = U_G @ D_G_inv_sqrt @ U_G.T
    
    C_outlier[i] = sp.linalg.svdvals(G_f_inv_sqrt @ H_f @ G_f_inv_sqrt)[0] / np.sqrt(1 + f_grad @ f_grad.T)
    '''
    
    #FI
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
        
        FI_outlier[i] =  nabla_f @ nabla_f.T            
        

    i = i + 1
    print(i)

       
'''
np.savetxt('MNIST_DenseNet121_128_best_setup0_part'+ str(set_part) + '_FI_n_outlier.txt', FI_n_outlier, delimiter=' ')
np.savetxt('MNIST_DenseNet121_128_best_setup0_part'+ str(set_part) + '_Jacob_n_outlier.txt', Jacob_n_outlier, delimiter=' ')
np.savetxt('MNIST_DenseNet121_128_best_setup0_part'+ str(set_part) + '_C_n_outlier.txt', C_n_outlier, delimiter=' ')


np.savetxt('MNIST_DenseNet121_128_best_setup0_part'+ str(set_part) + '_FI_outlier.txt', FI_outlier, delimiter=' ')
np.savetxt('MNIST_DenseNet121_128_best_setup0_part'+ str(set_part) + '_Jacob_outlier.txt', Jacob_outlier, delimiter=' ')
np.savetxt('MNIST_DenseNet121_128_best_setup0_part'+ str(set_part) + '_C_outlier.txt', C_outlier, delimiter=' ')
'''

     

np.savetxt('MNIST_DenseNet121_128_best_setup0_FI_n_outlier.txt', FI_n_outlier, delimiter=' ')
np.savetxt('MNIST_DenseNet121_128_best_setup0_Jacob_n_outlier.txt', Jacob_n_outlier, delimiter=' ')
#np.savetxt('MNIST_DenseNet121_128_best_setup0_C_n_outlier.txt', C_n_outlier, delimiter=' ')


np.savetxt('MNIST_DenseNet121_128_best_setup0_FI_outlier.txt', FI_outlier, delimiter=' ')
np.savetxt('MNIST_DenseNet121_128_best_setup0_Jacob_outlier.txt', Jacob_outlier, delimiter=' ')
#np.savetxt('MNIST_DenseNet121_128_best_setup0_C_outlier.txt', C_outlier, delimiter=' ')



