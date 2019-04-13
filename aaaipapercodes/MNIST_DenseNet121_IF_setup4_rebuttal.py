import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#import sys
#set_part = int(sys.argv[1])
import matplotlib.pyplot as plt
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




x_test = np.load('MNIST_test_img.npy')
y_test = np.load('MNIST_test_label.npy')


y_test_pred = np.load('MNIST_DenseNet121_test_pred_label.npy')


y_test_pred_prob = np.load('MNIST_DenseNet121_test_pred_prob.npy')

FI_test= np.loadtxt('setup3_MNIST_DenseNet121_test_manhattan.txt')


index = []
for i in range(y_test.shape[0]):
    if y_test[i] == y_test_pred[i] and FI_test[i,0]>0.6 and y_test_pred_prob[i]>0.5:
        index.append(i)
 
y_test[index]   
y_test_pred_prob[index]    
FI_test[index,0]
        
#max(y_test_pred_prob[index])#0.7040077
#id_maxFI=index[ np.argmax( y_test_pred_prob[index] ) ] #4860


#FI_test[id_maxFI,0]#0.420629054307938

#plt.imshow(x_test[id_maxFI,:,:,0],cmap="gray")


for i in range(19):
   plt.subplot(4, 5, i+1)
   plt.imshow(x_test[index[i],:,:,0],cmap="gray")
    


#lt.imshow(x_test[index[5],:,:,0],cmap="gray")
#y_test[index[5]]

#y_test[id_maxFI]

id_maxFI=index[4]


########################
#with tf.device('/cpu:0'):
model = load_model("training_lr1en4_MNIST_DenseNet121_epoch151to200_dropout_128_best.h5") 
    
    
###Setup 3: training and test sets wrt sum squared prob


grad_K=tf.concat(axis=1,values=[
   tf.concat(axis=0, values=[K.flatten(b)[...,None] for b in K.gradients(model.output[0,k], model.input)]) 
   for k in range(n_class)]
   )

#flatten(x)=tf.reshape(x, [-1])=np.reshape(x, [-1]); see https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py
            
iterate = K.function([model.input], [grad_K, model.output])  


#test
i = id_maxFI    

grad, pred_P = iterate([x_test[i][None]])


scale = [0,1,2,3] # kernel size=[3,5,7]

n_ix = x_test[i].shape[0]
n_iy = x_test[i].shape[1]

FI_test_pred=np.zeros((len(scale),n_ix,n_iy))


for k in range(len(scale)):    
    for i_x in range(n_ix):
        for i_y in range(n_iy):        
            nb_2d = np.array([[i_x+ii_x,i_y+ii_y] for ii_x in np.arange(max(-scale[k],-i_x),min(n_ix-i_x,scale[k]+1)) for ii_y in np.arange(max(-scale[k],-i_y),min(n_iy-i_y,scale[k]+1))]).T        
            nb_1d = np.ravel_multi_index(nb_2d,(n_ix,n_iy))
            
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
                
                FI_test_pred[k,i_x,i_y] =  nabla_f_pred @ nabla_f_pred.T

np.save('MNIST_DenseNet121_128_best_setup4_test_image'+ str(i) + '_multiscale_rebuttal', FI_test_pred)



