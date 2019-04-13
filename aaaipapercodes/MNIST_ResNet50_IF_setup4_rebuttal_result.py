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
x_test = np.load('MNIST_test_img.npy')
y_test = np.load('MNIST_test_label.npy')
y_test_pred = np.load('MNIST_ResNet50_test_pred_label.npy')
data_test= np.loadtxt('setup3_MNIST_ResNet50_test_manhattan.txt')


i=1878
y_test[i] #
y_test_pred[i] #


FI_test_pred = np.load('MNIST_ResNet50_128_best_setup4_test_image'+str(i)+'_multiscale_rebuttal.npy')

FI_test_pred[0,16,12]#0.9679657816886902

plt.subplot(1, 5, 1)
plt.imshow(x_test[i,:,:,0],cmap="gray")
plt.axis('off')
plt.clim(0, 255)


plt.subplot(1, 5, 2)
a=x_test[i,:,:,0].copy()
a[16,12]=x_test[i,16,12,0]-76
plt.imshow(a,cmap="gray")
plt.axis('off')
plt.clim(0, 255)

plt.subplot(1, 5, 3)
a=x_test[i,:,:,0].copy()
a[16,12]=x_test[i,16,12,0]-76
plt.imshow(a,cmap="gray")
plt.axis('off')
plt.clim(0, 255)


plt.subplot(1, 5, 4)
plt.imshow(FI_test_pred[0,:,:],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')


'''
plt.subplot(2, 3, 5)
b=FI_test_pred[0,:,:].copy()
b[16,12]=0
plt.imshow(b,cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')
'''


plt.subplot(1, 5, 5)
img1 = plt.imshow(FI_test_pred[0,:,:],cmap="nipy_spectral")
plt.clim(0, np.max(FI_test_pred));
img2 = plt.imshow(x_test[i,:,:,0],cmap="gray",alpha=0.6)
plt.show()
plt.axis('off')
#plt.title('Overlaid Scale-1 FI')



#get colorbar
plt.imshow(FI_test_pred[0,:,:],cmap="nipy_spectral")
plt.clim(0, np.max(FI_test_pred))
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20) 


