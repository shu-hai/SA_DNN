# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:28:02 2018

@author: HShu
"""
import matplotlib.pyplot as plt
import numpy as np

label = np.load('CIFAR10_train_label.npy')[:,0]
np.savetxt('CIFAR10_train_label.txt',label,delimiter=' ')

label = np.load('CIFAR10_test_label.npy')[:,0]
np.savetxt('CIFAR10_test_label.txt',label,delimiter=' ')
#######################
'''
Setup 1
'''
data= np.loadtxt('setup1_CIFAR10_ResNet50_manhattan.txt')
plt.ylim(ymax=10) 
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1),
           ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck'))
#plt.title('Manhattan plot of Setup 1')
x_train = np.load('CIFAR10_train_img.npy')
y_train = np.load('CIFAR10_train_label.npy')[:,0]
y_train_pred = np.load('CIFAR10_ResNet50_train_pred_label.npy')

#top 5 index = [16874 17762 28293 32169 38492]-1
i=38492
y_train[i -1]
y_train_pred[i-1]
data[i -1,0]#FI
plt.imshow(x_train[i -1])
plt.axis('off')

#######################
'''
Setup 2
'''
#local influence for all trainable parameters
data= np.loadtxt('setup2_CIFAR10_ResNet50_all_manhattan.txt')
plt.ylim(ymax=10) 
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1),
           ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck'))
#plt.title('Manhattan plot of Setup 2')

#local influence for each layer
data= np.loadtxt('setup2_CIFAR10_ResNet50_layers_manhattan.txt')
#fig, ax = plt.subplots()
#ax.set_color_cycle(['red', 'black'])
plt.xlim(xmin=-1,xmax=109)    
plt.ylim(ymax=10) 
for i in range(0,108):
    plt.plot(data[abs(data[:,1]-i)<1,1]+0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')

plt.xlabel('Trainable layer #')
plt.ylabel('FI')
#plt.title('Manhattan plot of Setup 2')


#######################
'''
Setup 3
'''
data_train= np.loadtxt('setup3_CIFAR10_ResNet50_train_manhattan.txt')
data_test= np.loadtxt('setup3_CIFAR10_ResNet50_test_manhattan.txt')

plt.ylim(ymax=3.7)  
for i in range(0,11):
    plt.plot(data_train[abs(data_train[:,1]-i)<1,1]-0.5,data_train[abs(data_train[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1),
           ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck'))


plt.ylim(ymax=3.7)  
for i in range(0,11):
    plt.plot(data_test[abs(data_test[:,1]-i)<1,1]-0.5,data_test[abs(data_test[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1),
           ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck'))

