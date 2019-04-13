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
y_test_pred = np.load('MNIST_DenseNet121_test_pred_label.npy')
data_test= np.loadtxt('setup3_MNIST_DenseNet121_test_manhattan.txt')


i=1878
y_test[i] #1
y_test_pred[i] #7
data_test[i,0]#FI=1.28

FI_test_pred = np.load('MNIST_DenseNet121_128_best_setup4_test_image'+str(i)+'_multiscale_rebuttal.npy')

#original
plt.imshow(x_test[i,:,:,0],cmap="gray")
plt.axis('off')

loc =np.unravel_index(np.argmax(FI_test_pred[0,:,:]),[28,28])
x_test[i,loc[0],loc[1],0]=255

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


