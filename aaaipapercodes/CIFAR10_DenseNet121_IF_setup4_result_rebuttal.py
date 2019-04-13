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
y_test_pred = np.load('CIFAR10_DenseNet121_test_pred_label.npy')
data_test= np.loadtxt('setup3_CIFAR10_DenseNet121_test_manhattan.txt')


i=2173-1
y_test[i] #5=dog
y_test_pred[i]#2=bird
data_test[i,0]#FI=2.22


FI_test_pred = np.load('./result/CIFAR10_DenseNet121_128_best_setup4_test_image'+str(i)+'_multiscale.npy')

#original
plt.imshow(x_test[i])
plt.axis('off')


#rgb
plt.subplot(3, 1, 1)
x_rgb=np.copy(x_test[i])
x_rgb[...,1]=0
x_rgb[...,2]=0
plt.imshow(x_rgb)
plt.axis('off')
#plt.colorbar()

plt.subplot(3, 1, 2)
x_rgb=np.copy(x_test[i])
x_rgb[...,0]=0
x_rgb[...,2]=0
plt.imshow(x_rgb)
plt.axis('off')
#plt.colorbar()

plt.subplot(3, 1, 3)
x_rgb=np.copy(x_test[i])
x_rgb[...,0]=0
x_rgb[...,1]=0
plt.imshow(x_rgb)
plt.axis('off')
#plt.colorbar()



#scale 1
plt.subplot(3, 1, 1)
plt.imshow(FI_test_pred[0,:,:,0],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')

plt.subplot(3, 1, 2)
plt.imshow(FI_test_pred[0,:,:,1],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')

plt.subplot(3, 1, 3)
plt.imshow(FI_test_pred[0,:,:,2],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')



#scale 3
plt.subplot(3, 1, 1)
plt.imshow(FI_test_pred[1,:,:,0],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')

plt.subplot(3, 1, 2)
plt.imshow(FI_test_pred[1,:,:,1],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')

plt.subplot(3, 1, 3)
plt.imshow(FI_test_pred[1,:,:,2],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')




#scale 5
plt.subplot(3, 1, 1)
plt.imshow(FI_test_pred[2,:,:,0],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')

plt.subplot(3, 1, 2)
plt.imshow(FI_test_pred[2,:,:,1],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')

plt.subplot(3, 1, 3)
plt.imshow(FI_test_pred[2,:,:,2],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')



#scale 7
plt.subplot(3, 1, 1)
plt.imshow(FI_test_pred[3,:,:,0],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')

plt.subplot(3, 1, 2)
plt.imshow(FI_test_pred[3,:,:,1],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')

plt.subplot(3, 1, 3)
plt.imshow(FI_test_pred[3,:,:,2],cmap="nipy_spectral")
plt.axis('off')
#plt.colorbar()
plt.clim(0, np.max(FI_test_pred))
#plt.title('Scale-1 FI')


#colorbar
plt.imshow(FI_test_pred[3,:,:,2],cmap="nipy_spectral")
plt.axis('off')
plt.clim(0, np.max(FI_test_pred))
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=20) 


