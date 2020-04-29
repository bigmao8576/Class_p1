
import os

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


def uni_model_train(df,model_info):
    OUTLIER_TH = model_info['OUTLIER_TH']
    DENSE_NUM = model_info['DENSE_NUM']
    LATENT_NUM = model_info['LATENT_NUM']
    LR = model_info['LR']
    MULTI_TASK = model_info['MULTI_TASK']
    BATCH_SIZE = model_info['BATCH_SIZE']
    EPOCHS = model_info['EPOCHS']
    
    
    
    path = 'report_nocw_uni_outlier_%0.4f_dens%d_latent%d_lr%0.6f'%(OUTLIER_TH,DENSE_NUM,LATENT_NUM,LR)
    if not os.path.exists(path):
        os.mkdir(path)
    
    plot_data ={'model_info':model_info
            }
    
    data_dic = utils.first_analysis_data(df,outlier_th=OUTLIER_TH)
    # remove outlier
    data_dic = utils.clean_data(data_dic,outlier_th=OUTLIER_TH)
    # input construction
    data_dic = utils.construction_in_out(data_dic)
    
    input_data, output_data = utils.get_input_out_array(data_dic,multi_task= MULTI_TASK)
    
    
    
    
            
    # now we get 10_fold
    fold_dic = utils.get_fold_dic(input_data,output_data)
    

    for fold_ind in range(1,11):
            
        fold = 'fold_%s'%fold_ind
        plot_data[fold] = {'loss':[],
                             'epoch':[],
                             'train_perf':[],
                             'val_perf':[]        
                             }
        
        train_input,train_label = fold_dic[fold]['input_train'],fold_dic[fold]['label_train']
        val_input,val_label = fold_dic[fold]['input_val'],fold_dic[fold]['label_val']
        cw = fold_dic[fold]['cw']
        
        model_para = {'dense_num':DENSE_NUM,
                      'latent_feature_num': LATENT_NUM}
         
            
        extractor = feature_extractor(train_input.shape[-1], model_para)
        classifier = output_classifier(model_para)
        
        input_ph = Input(shape=(train_input.shape[-1],))
        features = extractor(input_ph)
        model_output = classifier(features)
        
        model = Model(inputs=input_ph,outputs=model_output)
        
        
        
        model.compile(loss=weighted_categorical_crossentropy(np.array(cw)),
                      optimizer=Adam(lr=LR),
                      metrics=['accuracy'])
        
        
        
        
        for i in range(EPOCHS):
            batch_index = utils.shuffled_index(train_input.shape[0],BATCH_SIZE)
            for index in batch_index:
                batch_input = train_input[index]
                batch_label = train_label[index]

                _ = model.train_on_batch(batch_input, batch_label)
                #print(loss)
            
            if (i+1)%2==0:
                total_loss = model.evaluate(train_input,train_label)
                
                y_pred = model.predict(train_input)        
                train_perf = utils.perf(train_label,y_pred)
                
                y_pred = model.predict(val_input)
                val_perf = utils.perf(val_label,y_pred)
                
                print(model_info)
                print('Fold %s, Epoch %d, the training loss is %0.4f'%(fold,i,total_loss[0]))
                print('train_acc: %0.4f, recall: %0.4f, spe: %0.4f, auc: %0.4f'%(train_perf))
                print('val_acc: %0.4f, recall: %0.4f, spe: %0.4f, auc: %0.4f'%(val_perf))
                
                
                plot_data[fold]['epoch'].append(i)
                plot_data[fold]['loss'].append(total_loss[0])
                plot_data[fold]['train_perf'].append(train_perf)
                plot_data[fold]['val_perf'].append(val_perf)
                
                plot_utils.loss_curve(plot_data,fold,path,save = False)
            
            
        plot_utils.loss_curve(plot_data,fold,path,save = True)
        pickle.dump(plot_data, open( os.path.join(path,'plotdata.pkl'), "wb" ) )
    return plot_data