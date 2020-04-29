#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:11:17 2020

@author: bigmao
"""

import pandas as pd
import os
from sklearn.model_selection import KFold
import numpy as np
import utils
from class_model import feature_extractor, output_classifier,weighted_categorical_crossentropy
from keras.layers import Input
from keras.models import Model
import model_debug_tool as mdt
import matplotlib.pyplot as plt  
from keras.activations import softmax
import keras.backend as K
from keras.optimizers import Adam, SGD
from sklearn import metrics
import plot_utils
import pickle
from model_train_uni import uni_model_train

csv_path = 'All_data_noblanks_ante_intra.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    raise FileNotFoundError('The cvs file cannot be found')

#OUTLIER_TH = 0.000
MULTI_TASK = False
DENSE_NUM = 128
LATENT_NUM = 64
BATCH_SIZE = 400
EPOCHS = 200
LR = 0.0001


total_report = []

for OUTLIER_TH in [0.0,0.0001,0.0005,0.001,0.005,0.01]:
    for LATENT_NUM in [128]:

        model_info={'OUTLIER_TH': OUTLIER_TH,
                    'MULTI_TASK':MULTI_TASK,
                    'DENSE_NUM':DENSE_NUM,
                    'LATENT_NUM':LATENT_NUM,
                    'BATCH_SIZE':BATCH_SIZE,
                    'LR':LR,
                    'EPOCHS':EPOCHS}
        
        plot_data = uni_model_train(df,model_info)
        
        total_report.append(plot_data)
#plt.hist(a[:,5],bins=100);plt.show()
#a = model.layers[0].get_weights()


         #   array([0.80420073, 0.55704174])
         




















