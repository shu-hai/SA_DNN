# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:28:02 2018

@author: HShu
"""
import matplotlib.pyplot as plt
import numpy as np

label = np.load('MNIST_train_label.npy')
np.savetxt('MNIST_train_label.txt',label,delimiter=' ')

label = np.load('MNIST_test_label.npy')
np.savetxt('MNIST_test_label.txt',label,delimiter=' ')
#######################
'''
Setup 1
'''
data= np.loadtxt('setup1_MNIST_ResNet50_manhattan.txt')

plt.ylim(ymax=26)
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))

plt.ylim(ymax=1.01)
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))

#plt.title('Manhattan plot of Setup 1')
x_train = np.load('MNIST_train_img.npy')
y_train = np.load('MNIST_train_label.npy')
y_train_pred = np.load('MNIST_ResNet50_train_pred_label.npy')

#top 5 index = [1602  6054  9878 44140 50934]-1
i=50934
y_train[i -1]
y_train_pred[i-1]
data[i -1,0]
plt.imshow(x_train[i -1,:,:,0],cmap="gray")
plt.axis('off')
#######################
'''
Setup 2
'''
#local influence for all trainable parameters
data= np.loadtxt('setup2_MNIST_ResNet50_all_manhattan.txt')
plt.ylim(ymax=60)
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))


plt.ylim(ymax=1.01)
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))
#plt.title('Manhattan plot of Setup 2')


#local influence for each layer
data= np.loadtxt('setup2_MNIST_ResNet50_layers_manhattan.txt')
#fig, ax = plt.subplots()
#ax.set_color_cycle(['red', 'black'])
plt.xlim(xmin=-1,xmax=109)    
plt.ylim(ymax=26) 
for i in range(0,108):
    plt.plot(data[abs(data[:,1]-i)<1,1]+0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')

plt.xlabel('Trainable layer #')
plt.ylabel('FI')
#plt.title('Manhattan plot of Setup 2')

plt.xlim(xmin=-1,xmax=109)    
plt.ylim(ymax=1.01) 
for i in range(0,108):
    plt.plot(data[abs(data[:,1]-i)<1,1]+0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')

plt.xlabel('Trainable layer #')
plt.ylabel('FI')
#plt.title('Manhattan plot of Setup 2')


#######################
'''
Setup 3
'''
data_train= np.loadtxt('setup3_MNIST_ResNet50_train_manhattan.txt')
data_test= np.loadtxt('setup3_MNIST_ResNet50_test_manhattan.txt')

plt.ylim(ymax=3)
for i in range(0,11):
    plt.plot(data_train[abs(data_train[:,1]-i)<1,1]-0.5,data_train[abs(data_train[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))
#plt.title('Manhattan plot of Setup 3 training set')

plt.ylim(ymax=3)
for i in range(0,11):
    plt.plot(data_test[abs(data_test[:,1]-i)<1,1]-0.5,data_test[abs(data_test[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))
#plt.title('Manhattan plot of Setup 3 test set')


#######################
'''
Setup 4
'''
x_test = np.load('MNIST_test_img.npy')
y_test = np.load('MNIST_test_label.npy')
y_test_pred = np.load('MNIST_ResNet50_test_pred_label.npy')



FI_test_pred = np.load('./result/MNIST_ResNet50_128_best_setup4_test_image3073_multiscale.npy')

plt.subplot(1, 2, 1)
img1 = plt.imshow(x_test[3073,:,:,0],cmap="gray")
img2 = plt.imshow(FI_test_pred[0,:,:],cmap="nipy_spectral",alpha=0.6)
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(FI_test_pred[0,:,:],cmap="nipy_spectral")


plt.subplot(1, 5, 3)
plt.imshow(FI_test_pred[1,:,:],cmap="nipy_spectral")


plt.subplot(1, 5, 4)
plt.imshow(FI_test_pred[2,:,:],cmap="nipy_spectral")


plt.subplot(1, 5, 5)
plt.imshow(FI_test_pred[3,:,:],cmap="nipy_spectral")
#plt.colorbar()


#train
x_train = np.load('MNIST_train_img.npy')
y_train = np.load('MNIST_train_label.npy')
y_train_pred = np.load('MNIST_ResNet50_train_pred_label.npy')

FI_train = np.load('./result/MNIST_ResNet50_128_best_setup4_train_image23873_multiscale.npy')

plt.subplot(1, 2, 1)
img1 = plt.imshow(x_train[23873,:,:,0],cmap="gray")
img2 = plt.imshow(FI_train[0,:,:],cmap="nipy_spectral",alpha=0.6)
plt.show()

plt.subplot(1, 2, 2)
plt.imshow(FI_train[0,:,:],cmap="nipy_spectral")


plt.subplot(1, 5, 3)
plt.imshow(FI_test_pred[1,:,:],cmap="nipy_spectral")


plt.subplot(1, 5, 4)
plt.imshow(FI_test_pred[2,:,:],cmap="nipy_spectral")


plt.subplot(1, 5, 5)
plt.imshow(FI_test_pred[3,:,:],cmap="nipy_spectral")