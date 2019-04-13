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
data= np.loadtxt('setup1_MNIST_DenseNet121_manhattan.txt')

plt.ylim(ymax=1.01)
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))
#plt.title('Manhattan plot of Setup 1')
x_train = np.load('MNIST_train_img.npy')
y_train = np.load('MNIST_train_label.npy')
y_train_pred = np.load('MNIST_DenseNet121_train_pred_label.npy')

#top 5 index = [23874 31209 35746 41474 49160]-1
i=49160
y_train[i-1]
y_train_pred[i-1]
data[i -1,0]
plt.imshow(x_train[i-1,:,:,0],cmap="gray")
plt.axis('off')
#######################
'''
Setup 2
'''
#local influence for all trainable parameters
data= np.loadtxt('setup2_MNIST_DenseNet121_all_manhattan.txt')
plt.ylim(ymax=1.01)
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))
#plt.title('Manhattan plot of Setup 2')


#local influence for each layer
data= np.loadtxt('setup2_MNIST_DenseNet121_layers_manhattan.txt')
#fig, ax = plt.subplots()
#ax.set_color_cycle(['red', 'black'])
plt.xlim(xmin=-1,xmax=245)    
plt.ylim(ymax=1.01)
for i in range(0,243):
    plt.plot(data[abs(data[:,1]-i)<1,1]+0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')

plt.xlabel('Trainable layer #')
plt.ylabel('FI')
#plt.title('Manhattan plot of Setup 2')


#######################
'''
Setup 3
'''
data_train= np.loadtxt('setup3_MNIST_DenseNet121_train_manhattan.txt')
data_test= np.loadtxt('setup3_MNIST_DenseNet121_test_manhattan.txt')

plt.ylim(ymax=1.3)  
for i in range(0,11):
    plt.plot(data_train[abs(data_train[:,1]-i)<1,1]-0.5,data_train[abs(data_train[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))

plt.ylim(ymax=1.3)  
for i in range(0,11):
    plt.plot(data_test[abs(data_test[:,1]-i)<1,1]-0.5,data_test[abs(data_test[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1))

#top 5 index = [917 2370 3074 4741 9665]-1
x_test = np.load('MNIST_test_img.npy')
y_test = np.load('MNIST_test_label.npy')
y_test_pred = np.load('MNIST_DenseNet121_test_pred_label.npy')

y_test[2370-1]
y_test_pred[2370-1]
data_test[917 -1,0]
plt.imshow(x_test[917-1,:,:,0],cmap="gray")


#######################
'''
Setup 4
'''
x_test = np.load('MNIST_test_img.npy')
y_test = np.load('MNIST_test_label.npy')
y_test_pred = np.load('MNIST_DenseNet121_test_pred_label.npy')
data_test= np.loadtxt('setup3_MNIST_DenseNet121_test_manhattan.txt')


i=3074-1
y_test[i] #1
y_test_pred[i] #7
data_test[i,0]#FI=1.28

FI_test_pred = np.load('./result/MNIST_DenseNet121_128_best_setup4_test_image'+str(i)+'_multiscale.npy')

#original
plt.imshow(x_test[i,:,:,0],cmap="gray")
plt.axis('off')

#scale 1
plt.subplot(1, 2, 1)
plt.imshow(FI_test_pred[0,:,:],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')

plt.subplot(1, 2, 2)
img1 = plt.imshow(FI_test_pred[0,:,:],cmap="nipy_spectral")
plt.clim(0, np.max(FI_test_pred));
img2 = plt.imshow(x_test[i,:,:,0],cmap="gray",alpha=0.6)
plt.show()
plt.axis('off')
#plt.title('Overlaid Scale-1 FI')


#scale 3
plt.subplot(1, 2, 1)
plt.imshow(FI_test_pred[1,:,:],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))

plt.subplot(1, 2, 2)
img1 = plt.imshow(FI_test_pred[1,:,:],cmap="nipy_spectral")
plt.clim(0, np.max(FI_test_pred));
img2 = plt.imshow(x_test[i,:,:,0],cmap="gray",alpha=0.6)
plt.show()
plt.axis('off')



#scale 5
plt.subplot(1, 2, 1)
plt.imshow(FI_test_pred[2,:,:],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))

plt.subplot(1, 2, 2)
img1 = plt.imshow(FI_test_pred[2,:,:],cmap="nipy_spectral")
plt.clim(0, np.max(FI_test_pred));
img2 = plt.imshow(x_test[i,:,:,0],cmap="gray",alpha=0.6)
plt.show()
plt.axis('off')



#scale 7
plt.subplot(1, 2, 1)
plt.imshow(FI_test_pred[3,:,:],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))

plt.subplot(1, 2, 2)
img1 = plt.imshow(FI_test_pred[3,:,:],cmap="nipy_spectral")
plt.clim(0, np.max(FI_test_pred));
img2 = plt.imshow(x_test[i,:,:,0],cmap="gray",alpha=0.6)
plt.show()
plt.axis('off')

#get colorbar
plt.imshow(FI_test_pred[3,:,:],cmap="nipy_spectral")
plt.clim(0, np.max(FI_test_pred))
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=30) 


