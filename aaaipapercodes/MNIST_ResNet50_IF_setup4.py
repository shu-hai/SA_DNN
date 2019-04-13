import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#import sys
#set_part = int(sys.argv[1])

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

#x_train = np.load('MNIST_train_img.npy')
#y_train = np.load('MNIST_train_label.npy')
#y_train_pred = np.load('MNIST_ResNet50_train_pred_label.npy')# predicted label

x_test = np.load('MNIST_test_img.npy')
y_test_pred = np.load('MNIST_ResNet50_test_pred_label.npy')

#x perturbation
#with tf.device('/cpu:0'):
model = load_model("training_lr1en4_MNIST_ResNet50_epoch151to200_dropout_128_best.h5") 
    
    
###Setup 3: training and test sets wrt sum squared prob


grad_K=tf.concat(axis=1,values=[
   tf.concat(axis=0, values=[K.flatten(b)[...,None] for b in K.gradients(model.output[0,k], model.input)]) 
   for k in range(n_class)]
   )

#flatten(x)=tf.reshape(x, [-1])=np.reshape(x, [-1]); see https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
            
iterate = K.function([model.input], [grad_K, model.output])  

i =   

grad, pred_P = iterate([x_test[i][None]])

L0 = grad @ np.diag( ((pred_P[0,])**0.5+epsilon)**-1)

f_grad_pred = ( grad[:,y_test_pred[i]]/(pred_P[0,y_test_pred[i]]+epsilon) ).T
    
B0, D_L0, A0 = sp.linalg.svd(L0, full_matrices=False)
rank_L0 = sum(D_L0>epsilon)
rank_L0 

B0 = B0[:,:rank_L0]    
A0 = np.diag(D_L0[:rank_L0]) @ A0[:rank_L0,:]   

U_A, D_0, _ = sp.linalg.svd( A0 @ A0.T, full_matrices=True)
D_0_inv = np.diag(D_0**-1)
D_0_inv_sqrt = np.diag(D_0**-0.5)

U_0 = B0 @ U_A
                  
nabla_f_pred = f_grad_pred @ U_0 @ D_0_inv_sqrt


scale = [1,2,3] # kernel size=[3,5,7]

FI_test_pred=np.zeros((len(scale),) + x_test[i].shape[:2])

for k in range(len(scale)):    
    for i_x in range(x_test[i].shape[0]):
        for i_y in range(x_test[i].shape[1]):        
            nb_2d = np.array([[i_x+ii_x,i_y+ii_y] for ii_x in np.arange(max(-scale[k],-i_x),scale[k]+1) for ii_y in np.arange(max(-scale[k],-i_y),scale[k]+1)]).T        
            nb_1d = np.ravel_multi_index(nb_2d,x_test[i].shape[:2])
            FI_test_pred[k,i_x,i_y] =  nabla_f_pred[0,nb_1d] @ nabla_f_pred[0,nb_1d].T
       
np.savetxt('MNIST_ResNet50_128_best_setup4_image'+ str(i) + '_test.txt', FI_test_pred, delimiter=' ')




