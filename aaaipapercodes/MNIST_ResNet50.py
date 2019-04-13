import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import numpy as np
import tensorflow as tf

np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import random
random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'


import keras
from keras.models import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint,CSVLogger, Callback
from keras.utils.training_utils import multi_gpu_model

from keras import backend as K

from ResNet50 import ResNet50

#import time

class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)  


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)    
    
    
    
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        #pickle_dump(self.losses, "loss_history.pkl")


x_train = np.load('MNIST_train_img.npy')
y_train = keras.utils.to_categorical(np.load('MNIST_train_label.npy'), 10)

x_valid = np.load('MNIST_valid_img.npy')
y_valid = keras.utils.to_categorical(np.load('MNIST_valid_label.npy'), 10)

with tf.device('/cpu:0'):
    #model =  ResNet50(input_shape=x_train.shape[1:], classes=10)#Adam lr=1e-3 for 1 to 50 epochs
    #model=load_model('training_lr1en3_MNIST_ResNet50_epoch1to50_dropout_128.h5')#Adam lr=5e-4 for 51 to 100 epochs
    #model=load_model('training_lr5en4_MNIST_ResNet50_epoch51to100_dropout_128.h5')#Adam lr=2.5e-4 for 101 to 150 epochs
    model=load_model('training_lr2d5en4_MNIST_ResNet50_epoch101to150_dropout_128.h5')#Adam lr=1e-4 for 151 to 200 epochs


parallel_model = multi_gpu_model(model, gpus=2)

checkpoint = MultiGPUCheckpointCallback('training_lr1en4_MNIST_ResNet50_epoch151to200_dropout_128_best.h5',model, monitor='val_categorical_accuracy', verbose=2, save_best_only=True, mode='max')

logger = CSVLogger("training_lr1en4_MNIST_ResNet50_epoch151to200_dropout_128_best.log")
history = LossHistory()
time_callback = TimeHistory()

parallel_model.compile(optimizer = Adam(lr=1e-4), loss = 'categorical_crossentropy' , metrics = ['categorical_accuracy'])
parallel_model.fit(
    x=x_train,
    y=y_train, validation_data=(x_valid, y_valid),
    epochs=50, batch_size=128,
    verbose=2,
    callbacks=[checkpoint, logger,history])

#np.savetxt("time_training_lr1en3_lambda0d05_1to30_tumor"+tumortype+"_SS.txt", np.array([time_callback.times]), delimiter=' ')
#model.save('training_lr2d5en4_MNIST_ResNet50_epoch101to150_dropout_128.h5')

