import numpy as np
from scipy.ndimage.interpolation import shift
np.random.seed(0)
import random
random.seed(0)


y_img = np.load('MNIST_train_img.npy')
y_true_label = np.load('MNIST_train_label.npy')
y_DenseNet121_label = np.load('MNIST_DenseNet121_train_pred_label.npy')
y_DenseNet121_prob = np.load('MNIST_DenseNet121_train_pred_prob.npy')

y_ResNet50_label = np.load('MNIST_ResNet50_train_pred_label.npy')
y_ResNet50_prob = np.load('MNIST_ResNet50_train_pred_prob.npy')

inters = ((y_true_label-y_DenseNet121_label)==0) * ((y_true_label-y_ResNet50_label)==0)
inters0 = np.where(inters==0)

#y_prob = (y_DenseNet121_prob+y_ResNet50_prob)/2
#y_prob[inters0]=0

index270=np.zeros(270*10,dtype=int)
index30=np.zeros(30*10,dtype=int)

for k in range(10):
    index=np.where(y_true_label==k)[0]    
    index_temp = random.sample(list(set(index)-set(inters0[0])),300)
    index270[270*k:270*(k+1)] = index_temp[:270]
    index30[30*k:30*(k+1)] = index_temp[270:]
    
    

n_outlier_train = y_img[index270]
n_outlier_train_label = y_true_label[index270]

n_outlier_valid = y_img[index30]
n_outlier_valid_label = y_true_label[index30]


index_n300=list( (set(range(y_true_label.shape[0]))  - set(index270))- set(index30) )
index270_add = np.zeros(270*10,dtype=int)
index30_add = np.zeros(30*10,dtype=int)

for k in range(10):
    index_k=np.where(y_true_label==k)[0] 
    index_nk=list(set(index_n300)-set(index_k))
    index_temp=random.sample(index_nk,300)    
    index270_add[270*k:270*(k+1)] = index_temp[:270]
    index30_add[30*k:30*(k+1)] = index_temp[270:]
    
outlier_train_add = y_img[index270_add]  
outlier_valid_add = y_img[index30_add] 
  
 
outlier_train = np.zeros(n_outlier_train.shape)
outlier_valid = np.zeros(n_outlier_valid.shape)

for k in range(270*10):
    outlier_train[k,:,:,0] = np.maximum(shift(n_outlier_train[k,:,:,0],shift=np.random.choice(range(-4,5),2)),
                         shift(outlier_train_add[k,:,:,0],shift=np.random.choice(range(-4,5),2)))
    
for k in range(30*10):
    outlier_valid[k,:,:,0] = np.maximum(shift(n_outlier_valid[k,:,:,0],shift=np.random.choice(range(-4,5),2)),
                         shift(outlier_valid_add[k,:,:,0],shift=np.random.choice(range(-4,5),2)))
      
np.save('MNIST_train_img_Outlier',outlier_train)    
np.save('MNIST_train_label_Outlier',n_outlier_train_label)#same as nonoutliers.
np.savetxt('MNIST_train_label_Outlier.txt',n_outlier_train_label)

np.save('MNIST_valid_img_Outlier',outlier_valid)    
np.save('MNIST_valid_label_Outlier',n_outlier_valid_label)#same as nonoutliers.
np.savetxt('MNIST_valid_label_Outlier.txt',n_outlier_valid_label)



