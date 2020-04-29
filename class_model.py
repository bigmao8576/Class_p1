#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:21:19 2020

@author: bigmao
"""
from keras.layers.core import Dense
from keras.layers import LeakyReLU,BatchNormalization,Input,Add,Dropout
from keras.models import Model
from keras import backend as K

def res_block(inputs, model_para, last_out = None):
    
    '''
    if last out is none, that means this block is not used for final out put
    if last out is not none, one must feed a number for it as the final output channels.
    '''
    if last_out is None:
        last_neuron = model_para['dense_num']
    else:
        last_neuron = last_out
    
    out1 = Dense(model_para['dense_num'],
                 activation=None,
                 use_bias=False,
                 kernel_initializer='TruncatedNormal',
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None
                 )(inputs)
    out1 = Dropout(0.5)(out1)
    out1 = BatchNormalization()(out1)
    out1 = LeakyReLU(alpha=0.2)(out1)
    
    out2 = Dense(last_neuron,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='TruncatedNormal',
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None
                 )(out1)
    out2 = Dropout(0.5)(out2)
    out2 = BatchNormalization()(out2)
    project = Dense(last_neuron,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='TruncatedNormal',
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None
                 )(inputs)
    
    out3 = Add()([project,out2])
    final_out = LeakyReLU(alpha=0.2)(out3)
    
    return final_out



def feature_extractor(input_channel, model_para):
    inputs = Input(shape=(input_channel,))
    res1 = res_block(inputs,model_para)
    res2 = res_block(res1,model_para)
    res3 = res_block(res2,model_para,last_out = model_para['latent_feature_num'] )
    extractor = Model(inputs, res3)
    return extractor


def output_classifier(model_para,task='classification'):
    '''
    task can only be classification or regression
    '''
    
    if task == 'classification':
        last_func = 'softmax'
    elif task == 'regression':
        last_func = None
    else:
        raise ValueError('what is task??')
    
    
    inputs = Input(shape=(model_para['latent_feature_num'],))
    res1 = res_block(inputs,model_para)
    
    outputs = Dense(2,
                 activation=last_func,
                 use_bias=True,
                 kernel_initializer='TruncatedNormal',
                 bias_initializer='zeros',
                 kernel_regularizer=None, 
                 bias_regularizer=None
                 )(res1)
    #outputs = Dropout(0.5)(outputs)
    classifier = Model(inputs, outputs)
    return classifier


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss