import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf

np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import random
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train.shape
#(60000, 28, 28)

#x_test.shape
#(10000, 28, 28)

#normlized by mean and std
'''
for i in range(3):
    ch_mean = np.mean(x_train[:,:,:,i])
    ch_std = np.std(x_train[:,:,:,i])
    x_train[:,:,:,i] = (x_train[:,:,:,i]-ch_mean)/ch_std
    x_test[:,:,:,i] = (x_test[:,:,:,i]-ch_mean)/ch_std
'''
    
#Put 1/10 of training set as validation set

select=[]

for k in range(10):
    index = list(np.where(y_train==k)[0])
    select = select + random.sample(index,600)
    
x_valid = x_train[select]
y_valid = y_train[select]

notselect = np.setdiff1d(range(60000),select)

x_train = x_train[notselect]
y_train = y_train[notselect]


x_train = x_train[...,None]
x_valid = x_valid[...,None]
x_test = x_test[...,None]
    
np.save('MNIST_train_img',x_train)
np.save('MNIST_train_label',y_train)

np.save('MNIST_valid_img',x_valid)
np.save('MNIST_valid_label',y_valid)

np.save('MNIST_test_img',x_test)
np.save('MNIST_test_label',y_test)