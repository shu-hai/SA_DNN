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
data= np.loadtxt('setup1_CIFAR10_DenseNet121_manhattan.txt')
plt.ylim(ymax=80)
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1),
           ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck'))

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
y_train_pred = np.load('CIFAR10_DenseNet121_train_pred_label.npy')

#top 5 index = [3586  4314 32758 35956 43062]-1
i=43062
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
data= np.loadtxt('setup2_CIFAR10_DenseNet121_all_manhattan.txt')
plt.ylim(ymax=80)
for i in range(0,11):
    plt.plot(data[abs(data[:,1]-i)<1,1]-0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1),
           ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck'))

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
data= np.loadtxt('setup2_CIFAR10_DenseNet121_layers_manhattan.txt')
#fig, ax = plt.subplots()
#ax.set_color_cycle(['red', 'black'])
plt.xlim(xmin=-1,xmax=245)    
plt.ylim(ymax=80)
for i in range(0,243):
    plt.plot(data[abs(data[:,1]-i)<1,1]+0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')

plt.xlabel('Trainable layer #')
plt.ylabel('FI')
#plt.title('Manhattan plot of Setup 2')

plt.xlim(xmin=-1,xmax=245)    
plt.ylim(ymax=10)
for i in range(0,243):
    plt.plot(data[abs(data[:,1]-i)<1,1]+0.5,data[abs(data[:,1]-i)<1,0], ls='', marker='.')

plt.xlabel('Trainable layer #')
plt.ylabel('FI')

#######################
'''
Setup 3
'''
data_train= np.loadtxt('setup3_CIFAR10_DenseNet121_train_manhattan.txt')
data_test= np.loadtxt('setup3_CIFAR10_DenseNet121_test_manhattan.txt')

plt.ylim(ymax=2.6)  
for i in range(0,11):
    plt.plot(data_train[abs(data_train[:,1]-i)<1,1]-0.5,data_train[abs(data_train[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1),
           ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck'))


plt.ylim(ymax=2.6)  
for i in range(0,11):
    plt.plot(data_test[abs(data_test[:,1]-i)<1,1]-0.5,data_test[abs(data_test[:,1]-i)<1,0], ls='', marker='.')
plt.xlabel('Class')
plt.ylabel('FI')
plt.xticks(np.arange(0, 10, step=1),
           ('plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck'))

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


