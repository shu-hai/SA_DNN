import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#import sys
#set_part = int(sys.argv[1])

import numpy as np
import scipy as sp
import tensorflow as tf




from keras.models import *
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



x_train = np.load('CIFAR10_train_img.npy')
y_train = np.load('CIFAR10_train_label.npy')

x_test = np.load('CIFAR10_test_img.npy')
y_test_pred = np.load('CIFAR10_DenseNet121_test_pred_label.npy')


#with tf.device('/cpu:0'):
model = load_model("training_lr1en4_CIFAR10_DenseNet121_epoch401to500_dropout_128_best.h5") 
    
    
###Setup 3: training and test sets wrt sum squared prob


grad_K=tf.concat(axis=1,values=[
   tf.concat(axis=0, values=[K.flatten(b)[...,None] for b in K.gradients(model.output[0,k], model.input)]) 
   for k in range(n_class)]
   )

#flatten(x)=tf.reshape(x, [-1])=np.reshape(x, [-1]); see https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
            
iterate = K.function([model.input], [grad_K, model.output])  


#test
i = 2173-1   

grad, pred_P = iterate([x_test[i][None]])


scale = [0,1,2,3] # kernel size=[3,5,7]

n_ix = x_test[i].shape[0]
n_iy = x_test[i].shape[1]
n_ch = 3 # 3 channels: rgb

FI_test_pred=np.zeros((len(scale),n_ix,n_iy,n_ch))


for k in range(len(scale)):    
    for i_x in range(n_ix):
        for i_y in range(n_iy):        
            nb_2d = np.array([[i_x+ii_x,i_y+ii_y] for ii_x in np.arange(max(-scale[k],-i_x),min(n_ix-i_x,scale[k]+1)) for ii_y in np.arange(max(-scale[k],-i_y),min(n_iy-i_y,scale[k]+1))]).T        
            
            for i_ch in range(n_ch):     
                
                nb_2d_add = np.ones((1,nb_2d.shape[1]),dtype=int)*i_ch                
                nb_3d = np.concatenate((nb_2d,nb_2d_add),axis=0)
                
                nb_1d = np.ravel_multi_index(nb_3d,(n_ix,n_iy,n_ch))
                
                L0 = grad[nb_1d,:] @ np.diag( ((pred_P[0,])**0.5+epsilon)**-1)
    
                f_grad_pred = ( grad[nb_1d,y_test_pred[i]]/(pred_P[0,y_test_pred[i]]+epsilon) ).T
                    
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
                    
                    FI_test_pred[k,i_x,i_y,i_ch] =  nabla_f_pred @ nabla_f_pred.T

np.save('CIFAR10_DenseNet121_128_best_setup4_test_image'+ str(i) + '_multiscale', FI_test_pred)



