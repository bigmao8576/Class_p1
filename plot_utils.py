#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 21:56:07 2020

@author: bigmao
"""
import matplotlib.pyplot as plt  
import numpy as np
import os

def loss_curve(plotdata,fold,path,save = False):
    fold_data = plotdata[fold]
    train_perf = np.array(fold_data['train_perf'])
    val_perf = np.array(fold_data['val_perf'])
    
    loss_line, = plt.plot(fold_data['epoch'],fold_data['loss'],label='training_loss')
    tr_acc_line, = plt.plot(fold_data['epoch'],train_perf[:,0],label='train_acc')
    val_line, = plt.plot(fold_data['epoch'],val_perf[:,0],label='val_acc')
    plt.legend(handles=[loss_line,tr_acc_line,val_line])

    plt.xlabel('Epoch')
    plt.show()

    if save:
        plot_name = 'training_curve.png'
    
        plt.savefig(os.path.join(path,plot_name))
        plt.close()
