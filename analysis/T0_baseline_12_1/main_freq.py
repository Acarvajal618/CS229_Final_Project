# -*- coding: utf-8 -*-

import h5py
import sys
sys.path.insert(1, './../../src/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import plot_test as pt
import preprocessing as prep
import postprocessing as postp
import nn_model as nnm
import util
import os

noise_files = [ [ "./../../data/raw_data/chunk1.csv", "./../../data/raw_data/chunk1.hdf5"] ]
earthquake_files = [ ["./../../data/raw_data/chunk2.csv", "./../../data/raw_data/chunk2.hdf5"] ]

# train_data = exp_folder +'train_data_12.csv'
# train_labels = exp_folder +'train_labels_12.csv'

# test_data = exp_folder +'test_data_12.csv'
# test_labels = exp_folder +'test_labels_12.csv'

N_noise = 1
N_earthquake = 1
exp_folder = './../../data/e1_12_1/'

#Preprocessing names
filtered_csv = exp_folder + 'filtered_12.csv'
filtered_hdf5 = exp_folder + 'filtered_12.hdf5'
K_split = [.8,.1,.1]

train_csv_path = exp_folder + 'train.csv'
train_trace = train_csv_path
train_hdf5_path = exp_folder + 'train.hdf5'

val_csv_path = exp_folder + 'val.csv'
val_trace = val_csv_path
val_hdf5_path = exp_folder + 'val.hdf5'

test_csv_path = exp_folder + 'test.csv'
test_trace = test_csv_path
test_hdf5_path = exp_folder + 'test.hdf5'
transform_method = 'identity'

#Paths for training set/labels
xtrain = exp_folder + 'xtrain_freq.csv'
ytrain = exp_folder + 'ytrain_freq.csv'

#Paths for validation set/labels
xval = exp_folder + 'xval_freq.csv'
yval = exp_folder + 'yval_freq.csv'

#Paths for testing set and labels
xtest = exp_folder + 'xtest.csv'
ytest = exp_folder + 'ytest.csv'

train_analysis = './train_analysis.csv'
val_analysis = './val_analysis.csv'
test_analysis = './test_analysis.csv'

model_dir = './freq_model/'

def main():
    #Filter the HUGE datasets to a smaller chunk
    # prep.filter_merge_dataset(noise_files, earthquake_files, 
    #                 N_earthquake, N_noise, 
    #                 filtered_csv,filtered_hdf5)
    
    ##Create train, val, tests splits
    # prep.split(filtered_csv, filtered_hdf5, K_split, 
    #             train_csv_path,train_hdf5_path, 
    #             val_csv_path, val_hdf5_path,
    #             test_csv_path, test_hdf5_path)
    
    ##Transform the training and test data
    #prep.transform_data_pd( train_csv_path, train_hdf5_path, xtrain, ytrain, method = transform_method)
    #prep.transform_data_pd( val_csv_path, val_hdf5_path, xval, yval, method = transform_method)
    #prep.transform_data_pd( test_csv_path, test_hdf5_path, xtest, ytest, method = transform_method)
    
    #These arrays turned out to be unnecesssary but it's not worth the hassle to restructure the code to eliminate them
    s1_array = np.array([])
    s2_array = np.array([])
    s3_array = np.array([])
    l2_reg_array = np.array([])
    epoch_array = np.array([])
    af_array = np.array([])
    bs_array = np.array([])
    lr_array = np.array([])
    va_array = np.array([])
    ta_array = np.array([])
    iter_array = np.array([])

    #initialize values
    iter_to_opt = 0
    absolute_max_accuracy_obtained = 0
    abs_max_train_acc = 0
    total_loop_iter = 0
    loops_since_refresh = 0
    max_accuracy = 0
    train_acc_corresponding = 0
    
    #create folder to keep track of hyperparameter tuning data
    if not os.path.exists('./freq_hyp_tuning'):
            os.makedirs('./freq_hyp_tuning')   
    
    ###################### Hyperprarmeter tuning section. These values are frequently edited and fine-tuned ##################### 
    hyper_params = {"shape1": 1000, "shape2":771, "shape3": 0,"l2_reg": 0.0001, "epochs": 800,\
            "activation_function": 'tanh', "batch_size": 2048, "learning_rate": 0.00005 }

    #nested loops to sweep over different values of hyperparameters
    for i in range(2):
        for hyper_param in ["shape2","shape1","l2_reg","batch_size","shape3","learning_rate","activation_function", "epochs"]:
            if hyper_param == "shape1":
                param_value_array = [hyper_params[hyper_param],500,770,1000]
            elif hyper_param == "shape2":
                param_value_array = [hyper_params[hyper_param],500,770,1000]
            elif hyper_param == "shape3":
                param_value_array = [0,5]
            elif hyper_param == "activation_function":
                param_value_array = [hyper_params[hyper_param],'tanh', 'relu', 'sigmoid']
            elif hyper_param == "batch_size": 
                param_value_array = [hyper_params[hyper_param],512, 1024, 2048]
            elif hyper_param == "epochs":
                param_value_array = [hyper_params[hyper_param],500,800,1000]
            elif hyper_param == "learning_rate": 
                param_value_array = hyper_params["learning_rate"]*np.array([1,0.1,0.5, 2,10])
            elif hyper_param == "l2_reg": 
                param_value_array = np.array([0, 0.0001, 0.001])
    ###########################################################################################################################
    
    
            #This section is to save time by avoiding the redundancy of repeating the same set of hyperparameters    
            old_val = hyper_params[hyper_param]
            old_acc = max_accuracy
            old_train_acc = train_acc_corresponding
            max_accuracy = 0
            train_acc_corresponding = 0
            optimum_value = None
            for param_value in param_value_array:
                if old_val == param_value and old_acc > 0:
                    print(f'skipping repeat value of {old_val} with accuracy {old_acc} compared to current in_loop optimum {max_accuracy}')
                    if old_acc > max_accuracy:
                        print('reset optimum accuracy')
                        max_accuracy = old_acc
                        train_acc_corresponding = old_train_acc
                        optimum_value = param_value                    
                    continue

                hyper_params[hyper_param] = param_value
                model_dir = ''.join(['_'+str(a) + '_' + str(b)  for a,b in hyper_params.items()]) + '/'
                model_dir = './freq_model' + model_dir.replace('.', '_')
                total_loop_iter += 1
                loops_since_refresh += 1 
    
                training_accuracy, validation_accuracy = nnm.train(  xtrain,
                    ytrain,
                    xval,
                    yval,
                    [hyper_params["shape1"], hyper_params["shape2"], hyper_params["shape3"]] if hyper_params["shape3"] else [hyper_params["shape1"], hyper_params["shape2"]],
                    hyper_params["activation_function"],
                    hyper_params["l2_reg"],
                    hyper_params["epochs"],
                    hyper_params["batch_size"],
                    hyper_params["learning_rate"],
                    train_analysis,
                    train_trace,
                    val_analysis,
                    val_trace,
                    model_dir,
                    'testing')
                print(f'parameter {hyper_param} with value: {param_value}, iteration: {i}')
               
                #add values to vectors
                va_array = np.append(va_array,validation_accuracy)
                ta_array = np.append(ta_array,training_accuracy)
                s3_array = np.append(s3_array,hyper_params["shape3"])
                s2_array = np.append(s2_array,hyper_params["shape2"])
                s1_array = np.append(s1_array,hyper_params["shape1"])
                l2_reg_array = np.append(l2_reg_array,hyper_params["l2_reg"])
                af_array = np.append(af_array,hyper_params["activation_function"])
                bs_array = np.append(bs_array,hyper_params["batch_size"])
                lr_array = np.append(lr_array,hyper_params["learning_rate"])
                epoch_array = np.append(epoch_array,hyper_params["epochs"])
                iter_array = np.append(iter_array,total_loop_iter )

                
                ##Initially we were going to write to the csv only every 10 loops or so, but each loop take so long
                ##  we just decided to write afterevery loop
                #if loops_since_refresh > 5:
                if True:
                    loops_since_refresh = 0
                    
                    d = {"iternum": iter_array,"val_accuracy": va_array , "train_accuracy": ta_array, "shape3": s3_array,"shape2": s2_array, "shape1": s1_array, "l2_reg": l2_reg_array, "activation_function": af_array, "batch_size":bs_array, "learning_rate":lr_array, "epochs": epoch_array}
                    hp_df = pd.DataFrame(data = d)
                    hp_df.to_csv( './freq_hyp_tuning/hyperparameter_training_test.csv', index = True, mode = 'a')
                    
                    va_array = np.array([])
                    ta_array = np.array([])
                    s3_array = np.array([])
                    s2_array = np.array([])
                    s1_array = np.array([])
                    l2_reg_array = np.array([])
                    af_array = np.array([])
                    bs_array = np.array([])
                    lr_array = np.array([])
                    epoch_array = np.array([])
                    iter_array = np.array([])
                
                #update max acuracy values for just this parameter
                if validation_accuracy > max_accuracy:
                    max_accuracy = validation_accuracy
                    train_acc_corresponding = training_accuracy
                    optimum_value = param_value
                
                #update max accuracy over all loops thus far
                if validation_accuracy > absolute_max_accuracy_obtained:
                    absolute_max_accuracy_obtained = validation_accuracy
                    abs_max_train_acc = training_accuracy
                    iter_to_opt = i
                    absolute_optimum_parameters = hyper_params.copy()
                    print(f"this is max accuracy: {validation_accuracy}")
                    f = open('./freq_hyp_tuning/abs_opt_hyper_param.txt',"w")
                    f.write(f'At loop # {total_loop_iter}, Max Validation accuracy of {absolute_max_accuracy_obtained} and train accuracy of {abs_max_train_acc} occured at ' +  str(absolute_optimum_parameters) )
                    f.close()

            hyper_params[hyper_param] = optimum_value
   
    #once training is done, test on optimal model
    aop = absolute_optimum_parameters.copy()
    model_dir = './freq_best_model/'
    training_accuracy, validation_accuracy = nnm.train(  xtrain,
                    ytrain,
                    xval,
                    yval,
                    [aop["shape1"], aop["shape2"], aop["shape3"]] if aop["shape3"] else [aop["shape1"], aop["shape2"]],
                    aop["activation_function"],
                    aop["l2_reg"],
                    aop["epochs"],
                    aop["batch_size"],
                    aop["learning_rate"],
                    train_analysis,
                    train_trace,
                    val_analysis,
                    val_trace,
                    model_dir,
                    'testing')
    print(f'Max Validation accuracy of {validation_accuracy} and train accuracy of {training_accuracy} occured at ' +  str(aop) )

    test_acc = nnm.test(xtest,
               ytest, 
               test_analysis,
               test_trace,
               model_dir)
   
    ##Post processing
    # postp.error_analysis(predictions, test_labels, error_stats)
    # pt.plot_test(filtered_csv,filtered_hdf5, True)
    # pt.earthquake_test(filtered_csv,filtered_hdf5)
    # pt.noise_test(filtered_csv,filtered_hdf5)    


if __name__ == '__main__':
    main()
        
