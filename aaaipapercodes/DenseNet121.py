#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#modified from https://github.com/shu-hai/pytorch-cifar/blob/master/models/densenet.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from keras import backend as K
from keras.models import Model
from keras.layers import Activation, AveragePooling2D, BatchNormalization, Concatenate, Conv2D, Dense, Input, Dropout, GlobalAveragePooling2D 


from keras import regularizers


def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1, use_bias=True, kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(1e-4),
               name=name + '_conv')(x)
    x = Dropout(0.2)(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=True, kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(1e-4), 
                name=name + '_1_conv')(x1)
    x1 = Dropout(0.2)(x1)
    x1 = BatchNormalization(axis=bn_axis, name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=True, kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(1e-4),
                name=name + '_2_conv')(x1)
    x1 = Dropout(0.2)(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet121(input_shape=None, classes=10, blocks=[6, 12, 24, 16]):


    img_input = Input(shape=input_shape) 

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = Conv2D(64, (3, 3), kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(1e-4), strides=(1, 1), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(axis=bn_axis, name='bn')(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='fc'+str(classes))(x)


    # Create model.
    model = Model(img_input, x, name='densenet121')

    return model

