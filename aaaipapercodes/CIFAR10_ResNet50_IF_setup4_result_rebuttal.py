# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 16:28:02 2018

@author: HShu
"""
import matplotlib.pyplot as plt
import numpy as np


'''
Setup 4
'''
x_test = np.load('CIFAR10_test_img.npy')
y_test = np.load('CIFAR10_test_label.npy')
y_test_pred = np.load('CIFAR10_ResNet50_test_pred_label.npy')
data_test= np.loadtxt('setup3_CIFAR10_ResNet50_test_manhattan.txt')


i=7908
y_test[i] #5=dog
y_test_pred[i]#2=bird
data_test[i,0]#FI=2.22


FI_test_pred = np.load('CIFAR10_ResNet50_128_best_setup4_test_image'+str(i)+'_multiscale_rebuttal.npy')

#original
plt.imshow(x_test[i])
plt.axis('off')


FI_test_pred3=(FI_test_pred[0,:,:,0]+FI_test_pred[0,:,:,1]+FI_test_pred[0,:,:,2])/3
np.max(FI_test_pred3)#0.9884220560391744
np.unravel_index(np.argmax(FI_test_pred3),(32,32))

a=np.zeros((32,32))
a[4,13]=255
plt.imshow(a,cmap="gray")

x_test[i,4,13,:]


plt.subplot(1, 5, 1)
plt.imshow(x_test[i])
plt.axis('off')


plt.subplot(1, 5, 2)
a=x_test[i].copy()
a[4,13,:]=[0,0,0]
plt.imshow(a)
plt.axis('off')


plt.subplot(1,5, 3)
a=x_test[i].copy()
a[4,13,:]=[0,0,0]
plt.imshow(a)
plt.axis('off')


plt.subplot(1, 5, 4)
plt.imshow(FI_test_pred3,cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred3))
#plt.title('Scale-1 FI')
#cbar=plt.colorbar()
#cbar.ax.tick_params(labelsize=20)


plt.subplot(1,5, 5)
img1 = plt.imshow(FI_test_pred3,cmap="nipy_spectral")
plt.clim(0, np.max(FI_test_pred3));
img2 = plt.imshow(x_test[i],alpha=0.6)
plt.show()
plt.axis('off')
#plt.title('Overlaid Scale-1 FI')




#colorbar
plt.imshow(FI_test_pred3,cmap="nipy_spectral")
plt.axis('off')
plt.clim(0, np.max(FI_test_pred3))
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20) 


