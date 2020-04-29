#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:14:52 2020

@author: bigmao
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn import metrics


def list_type(temp_data):
    '''
    This function is used for checking the variables type.
    Say a column of data might have different types, this function
    can find all the types.
    if all the data in one column only have one data type, this function return the type
    
    in this project, it should be 'int' and 'float'
    '''
    ls = [type(item).__name__ for item in temp_data]
    var_type =  list(dict.fromkeys(ls))
    if len(var_type) == 1:
        return var_type[0]
    else:
        raise TypeError('Two types found!')

def labe2dic(temp_data):
    '''
    This function is converting the data into a dictionary
    
    '''
    digits =  sorted(list(dict.fromkeys(temp_data)))

    dic = {str(item):0 for item in digits}
    for item in temp_data:
        dic[str(item)] +=1
        
    return dic

def binary_ratio(temp_data):
    '''
    if a lable is binary, we need to examine the label ratio,this function is used to calculate the ratio
    
    '''
    temp = []
    dic = labe2dic(temp_data)
    for key in dic.keys():
        temp.append(dic[key])
    temp_ratio = min(temp)/sum(temp)
    if temp_ratio >1:
        temp_ratio = 1/temp_ratio
        
    return temp_ratio

def first_analysis_data(df,outlier_th=0.005):
    
    '''
    This is the first round of analyzing data
    '''
    print('Now I am doing some initial analysis on the data')
    data_dic = {}

    
    for col in df.columns:
        temp_data = df[col].to_list()
        
        # check whether there is a constant
        if min(temp_data) == max(temp_data):
            raise ValueError('the variable %s is a constant! '%col)
        
        data_dic[col] = {'ori_data':temp_data}
        data_dic[col]['value_type'] = list_type(temp_data)
        data_dic[col]['notes'] = ''
        if col == 'Complicated_delivery_2':
            data_dic[col]['in_or_out'] = 'out'
        else:
            data_dic[col]['in_or_out']  = 'in'
            
        
        if list_type(temp_data)  == 'int':
        # discrete value
            data_dic[col]['data_dis'] = labe2dic(temp_data)
            if len(dict.fromkeys(temp_data))==2:
                # should be binary classification
                data_dic[col]['class_type']='binary'
                data_dic[col]['data_ratio'] = binary_ratio(temp_data)
                
                if max(temp_data)==1 and min(temp_data) ==0: # the label is just 0 and 1
                    if data_dic[col]['data_ratio']>0.05:
                        data_dic[col]['available_for_multi']=True
                        
                    else:
                        data_dic[col]['available_for_multi']=False
                        data_dic[col]['notes'] +='It is binary class, but the data is seriously unblanced.'
                        
                else: # the label might be something else, but still binary
                    data_dic[col]['notes'] += 'The binary class is not start from zero.'
                    if data_dic[col]['data_ratio']>0.05:
                        data_dic[col]['available_for_multi']=True
                        
                    else:
                        data_dic[col]['available_for_multi']=False
                        data_dic[col]['notes']+='It is binary class, but the data is seriously unblanced.' 
                if data_dic[col]['data_ratio']<outlier_th:    
                        data_dic[col]['notes']+='The ratio of sample number is too low, which is %0.4f, consider outlier removal and not to use this variable'%(data_dic[col]['data_ratio'])
            else:
                # multiple classification
                data_dic[col]['class_type']='multiple'
                
                # order the sample numbers in each class
                temp_list = [data_dic[col]['data_dis'][item] for item in data_dic[col]['data_dis'].keys()]
                temp_list = sorted(temp_list,reverse=True)
                
                
                if temp_list[0]/sum(temp_list)>0.9:
                    data_dic[col]['available_for_multi']=False
                    data_dic[col]['notes']+='The sample number of major class exceeds 90%% of total sample number, which is %0.4f.'%(temp_list[0]/sum(temp_list))
                elif temp_list[1]/temp_list[0] < 0.05:
                    data_dic[col]['available_for_multi']=False
                    data_dic[col]['notes'] +='The sample number of major class is not too large, but the data is seriously unblanced.'
                    print(col)
                else:
                    data_dic[col]['available_for_multi']=True
                    
                    if (temp_list[0]+temp_list[1])/sum(temp_list)>0.95:
                        data_dic[col]['notes'] +='The most minor two class has samples exceeds 95%%, which is %0.4f, consider mask.'%((temp_list[0]+temp_list[1])/sum(temp_list))
                    if len(dict.fromkeys(temp_data))==(max(temp_data)-min(temp_data)+1):
                    # that means the label indexes are continuous integers  
                        data_dic[col]['notes'] +='Continuous integers, consider additional regression task'
                if temp_list[-1]<outlier_th*sum(temp_list):
                        data_dic[col]['notes'] +='The most minor class has samples smaller than %d, consider outlier removal.'%(int(0.005*sum(temp_list)))
        else:
            data_dic[col]['class_type']='regression'
            data_dic[col]['available_for_multi']=True
            cont_data = temp_data
            
            mean = np.mean(cont_data)
            std = np.std(cont_data)
            data_dic[col]['mean_and_std']=[mean,std]
            data_dic[col]['3std_bound']=[mean-3*std,mean+3*std]
            if min(cont_data)< mean-4*std or max(cont_data)> mean+4*std:
                if outlier_th!=0.0:
                    data_dic[col]['notes'] +='some data exceeds mean +/- 4std, consider outlier removal.'
            print('Initial analysis completed')
    return data_dic

def getIndexPositions(listOfElements, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    indexPosList = [i for i in range(len(listOfElements)) if listOfElements[i]==element]
 
    return indexPosList

def find_outlier_index(data_dic,outlier_th=0.005):
    total_outlier_index = []
    for col in data_dic.keys():
        
        temp_data = data_dic[col]['ori_data']
        temp_note = data_dic[col]['notes']
        if data_dic[col]['class_type']=='binary':
            if 'consider outlier removal and not to use this variable' in temp_note:
                temp_dis = data_dic[col]['data_dis']
                temp = min(temp_dis.values()) 
                # find out the label with small samples
                res = [key for key in temp_dis if temp_dis[key] == temp]
                outlier_index = getIndexPositions(temp_data,int(res[0]))
                #double check
                if len(outlier_index)!= temp:
                    raise ValueError('Be careful! the minimun sample number does not match the outlier numbers')
                    
                total_outlier_index.append(outlier_index)
        elif data_dic[col]['class_type']=='multiple':
            if 'consider outlier removal' in temp_note:
                temp_dis = data_dic[col]['data_dis']
    
                # find out the label with small samples
                res = [key for key in temp_dis if temp_dis[key] <outlier_th*len(temp_data)]
                outlier_index = []
                for item in res:
                    temp = getIndexPositions(temp_data,int(item))
                    outlier_index +=  temp
                total_outlier_index.append(outlier_index)
    
        else:
            if 'consider outlier removal' in temp_note:
                mean,std = data_dic[col]['mean_and_std']
                indexPosList = [i for i in range(len(temp_data)) if temp_data[i]< mean-4*std or temp_data[i]> mean+4*std]
                total_outlier_index.append(indexPosList)
    
    temp = []
    for item in total_outlier_index:
        temp += item
        
    total_outlier_index = set(temp)
    total_outlier_index = list(total_outlier_index)
    
    return sorted(total_outlier_index)

def one_hot(temp_data):
    if type(temp_data).__name__ == 'list':
        temp_data = np.array(temp_data)
        temp_data = np.expand_dims(temp_data,1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(temp_data)
        temp_data = enc.transform(temp_data).toarray()
        return np.float32(temp_data)
    else:
        raise TypeError('The input is not a list')
        
def class_weight(one_hot_data):
    '''
    This funciton is used for calculating the weight for imbalanced classes
    '''
    
    sample_number = np.sum(one_hot_data,0) # how many samples in each class
    N = np.sum(sample_number) # how many total samples
    C = len(sample_number) # how many class
    w = N/sample_number
    w = w/C
    return w

def clean_data(data_dic,outlier_th):
        
    # now find outlier 
    outlier_index = find_outlier_index(data_dic,outlier_th=outlier_th)
    
    print('%d outlier will be removed'%len(outlier_index))
    unavaliable_input = []
    for col in data_dic.keys():
        temp_data = data_dic[col]['ori_data']
        
        clean_data = [temp_data[i] for i in range(len(temp_data)) if i not in outlier_index]
        data_dic[col]['clean_data'] = {'raw_data':clean_data}
        data_dic[col]['clean_data']['notes'] = ''
        
        if data_dic[col]['value_type'] == 'int':
            data_dic[col]['clean_data']['weight_after_ol'] = labe2dic(clean_data)
            if len(labe2dic(clean_data)) == 1:
                
                data_dic[col]['clean_data']['notes'] += 'ATTENTION: After outlier removal, the variable %s only has one values %s, will not be considered as input variable'%(col,list(labe2dic(clean_data).keys())[0])
                print(data_dic[col]['clean_data']['notes'])
                data_dic[col]['clean_data']['available_for_input'] = False
                unavaliable_input.append(col)
            elif len(labe2dic(clean_data)) == 2:
                data_dic[col]['clean_data']['class_type'] = 'binary'
                data_dic[col]['clean_data']['available_for_input'] = True
                if len(labe2dic(temp_data)) != 2:
                    data_dic[col]['clean_data']['notes'] += 'ATTENTION: After outlier removal, the variable %s changed from multiple to binary integer'%(col)
                    print(data_dic[col]['clean_data']['notes'])
                
            else:
                data_dic[col]['clean_data']['class_type'] = 'multiple'
                data_dic[col]['clean_data']['available_for_input'] = True
        else: # continuous
                data_dic[col]['clean_data']['class_type'] = 'regression'
                data_dic[col]['clean_data']['available_for_input'] = True
    
    if  len(outlier_index) ==0:
        print('No outlier was removed')
    else:
      
        if unavaliable_input!=[]:
            print('After outlier removal, the following variables wont be used')
            print(unavaliable_input)
        else:
            
            print('After outlier removal, all the variables will be used')
    return data_dic

def construction_in_out(data_dic):
    '''
    The input and output may not be suitable for the model.
    binary input: no need for one-hot, but need to convert list to (N,1) array with float32
    binary output: need one_hot, and convert list to (N,2) array with float32
    
    multiple input: one_hot, and convert list to (N, C) array with float32
    multiple output: same to multiple input
    
    continuous value, to float32, normalization would be used afer 10-fold
    '''
    for col in data_dic.keys(): 
    
    # binary is already 1 and 0, we don't need to consider
        if data_dic[col]['clean_data']['available_for_input']:
            temp_data = data_dic[col]['clean_data']['raw_data']
            if data_dic[col]['clean_data']['class_type'] == 'binary':
                 
                

                data_dic[col]['clean_data']['output_data'] = one_hot(temp_data)
                data_dic[col]['clean_data']['class_weight'] = class_weight(data_dic[col]['clean_data']['output_data'])
                temp_data = np.array(temp_data)
                temp_data = np.expand_dims(temp_data,-1)
                data_dic[col]['clean_data']['input_data'] = np.float32(temp_data)
    
                
            elif data_dic[col]['clean_data']['class_type'] == 'multiple':
                
                data_dic[col]['clean_data']['input_data'] = one_hot(temp_data)
                data_dic[col]['clean_data']['output_data'] = data_dic[col]['clean_data']['input_data']
                data_dic[col]['clean_data']['class_weight'] = class_weight(data_dic[col]['clean_data']['input_data'])
               # break
                #print(col)
            #data_dic[col]['class_type']='multiple'
                #temp_data = data_dic[col]['clean_data']['raw_data'] 
            else:
                temp_data = data_dic[col]['clean_data']['raw_data']
                temp_data = np.array(temp_data)
                temp_data = np.expand_dims(temp_data,-1)
                data_dic[col]['clean_data']['input_data'] = np.float32(temp_data)
                data_dic[col]['clean_data']['output_data'] = np.float32(temp_data)

    return data_dic

def get_input_out_array(data_dic,multi_task= False,debug = False):
    '''
    if multi_task is true, the output array is a list,
    if multi_task is false, the output array is a numpy array
    '''

    input_list = []
    for col in data_dic.keys(): 
        if data_dic[col]['clean_data']['available_for_input'] and data_dic[col]['in_or_out']  == 'in':
            input_list.append(data_dic[col]['clean_data']['input_data'])

    for i in range(len(input_list)):
        if i==0:
            input_array = input_list[i]
        else:
            input_array = np.concatenate([input_array,input_list[i]],axis =-1)
    
    # now let's get the output
    
    for col in data_dic.keys(): 
        if data_dic[col]['in_or_out']  == 'out':
            output_array = data_dic[col]['clean_data']['output_data']
            output_name = [col]
    
    if multi_task:
        output_array = [output_array]
        
        for col in data_dic.keys(): 
            if data_dic[col]['in_or_out']  == 'in' and data_dic[col]['clean_data']['available_for_input']:
                temp_output = data_dic[col]['clean_data']['output_data']  
                
                data_exam = np.sum(temp_output,0)
                if np.min(data_exam) > 100:
                    output_array.append(temp_output)
                    output_name.append(col)
    if debug:
        return input_array,output_array,output_name
    else:
            
        return input_array,output_array

def get_fold_dic(input_data,output_data):
    print('Now we get 10_fold')
    fold_dic = {}
    kf = KFold(n_splits=10,shuffle=True)  
    
    fold_num = 0 
    for train_index, val_index in kf.split(input_data):
        fold_num +=1
        key_name = 'fold_%s'%str(fold_num)

        
        input_train = input_data[train_index]   
        input_val = input_data[val_index]
        
        # normalization
    
        for i in range(input_train.shape[1]):
            temp = input_train[:,i]
            if len(labe2dic(temp))>2:
    
                mean = np.mean(temp)
                std = np.std(temp)

                input_train[:,i] = (temp-mean)/std
                

                temp_val = input_val[:,i]
                input_val[:,i] = (temp_val-mean)/std

                
        fold_dic[key_name] = {'input_train':input_train,
                              'input_val':input_val                                                  
                              }
        
        #now let's deal with the output
        
        if type(output_data).__name__ =='ndarray': # one task
            label_train = output_data[train_index]   
            label_val = output_data[val_index] 
            cw = class_weight(label_train)
            fold_dic[key_name]['label_train']=label_train
            fold_dic[key_name]['label_val']=label_val
            fold_dic[key_name]['cw']=cw
                                        
        
        elif type(output_data).__name__ =='list': # multi-task
            label_train =[]
            label_val = []
            cw = []
            for item in output_data:
                
                temp_label_train = item[train_index]
                temp_label_val = item[val_index]
                
                if item.shape[1] !=1: # discrete
                    label_train.append(temp_label_train)
                    label_val.append(temp_label_val)
                    
                    cw.append(class_weight(temp_label_train))
                else: # continuous
                    
                    
                    mean = np.mean(temp_label_train)
                    std = np.std(temp_label_train)
                    temp_label_train = (temp_label_train-mean)/std
                    temp_label_val = (temp_label_val-mean)/std
    
                    label_train.append(temp_label_train)
                    label_val.append(temp_label_val)
                    cw.append([]) # occupy a position
            fold_dic[key_name]['label_train']=label_train
            fold_dic[key_name]['label_val']=label_val
            fold_dic[key_name]['cw']=cw 
        else:
            raise TypeError('unknown output type!')
    return fold_dic

def data_index_list(length,batch_size):
    index_list = []
    epochs = (length//batch_size)+1
    for i in range(epochs):
        if i <epochs-1:
        
            index_list.append([i*batch_size,i*batch_size+batch_size])
        else:

            index_list.append([i*batch_size,length])
            
    return index_list

def shuffled_index(length,batch_size):
    
    index_list = []
    
    psuedo_index = [i for i in range(length)]
    np.random.shuffle(psuedo_index)
    batch_index  = data_index_list(length,batch_size)
    for i in batch_index:
        index_list.append(psuedo_index[i[0]:i[1]])
    return index_list

def auc(y_true,y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr)

def perf(y_true,y_pred):
    
    y_pred = y_pred[:,1]
    y_pred_dig = np.round(y_pred)
    y_true = y_true[:,1]
    
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_dig).ravel()
    
    
    acc = (tp+tn)/(tn+fp+fn+tp)
    recall = tp/(tp+fn)
    speci = tn/(tn+fp)
    
    return acc, recall, speci, auc(y_true,y_pred)