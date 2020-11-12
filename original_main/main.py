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

noise_files = [ [ "./../../data/raw_data/chunk1.csv", "./../../data/raw_data/chunk1.hdf5"] ]
earthquake_files = [ ["./../../data/raw_data/chunk2.csv", "./../../data/raw_data/chunk2.hdf5"] ]

# train_data = exp_folder +'train_data_12.csv'
# train_labels = exp_folder +'train_labels_12.csv'

# test_data = exp_folder +'test_data_12.csv'
# test_labels = exp_folder +'test_labels_12.csv'

N_noise = .1
N_earthquake = .1
exp_folder = './../../data_testing/'

#Preprocessing names
# filtered_csv = exp_folder + 'filtered_12.csv'
# filtered_hdf5 = exp_folder + 'filtered_12.hdf5'
# K_split = [.8,.1,.1]

# train_csv_path = exp_folder + 'train.csv'
# train_hdf5_path = exp_folder + 'train.hdf5'

# val_csv_path = exp_folder + 'val.csv'
# val_hdf5_path = exp_folder + 'val.hdf5'

# test_csv_path = exp_folder + 'test.csv'
# test_hdf5_path = exp_folder + 'test.hdf5'
# transform_method = 'identity'

#Paths for training set/labels
xtrain = exp_folder + 'xtrain.csv'
ytrain = exp_folder + 'ytrain.csv'

#Paths for validation set/labels
xval = exp_folder + 'xval.csv'
yval = exp_folder + 'yval.csv'

#Paths for testing set and labels
xtest = exp_folder + 'xtest.csv'
ytest = exp_folder + 'ytest.csv'

train_analysis = './results/train_analysis.csv'
val_analysis = './results/val_analysis.csv'
test_analysis = './results/test_analysis.csv'

model_dir = './model/'

shape = [10,10]
l2_reg = 0
epochs = 10
activation_function = 'relu'
batch_size = 10
learning_rate = .01

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
    
    #Transform the training and test data
    # prep.transform_data_pd( train_csv_path, train_hdf5_path, xtrain, ytrain, method = transform_method)
    # prep.transform_data_pd( val_csv_path, val_hdf5_path, xval, yval, method = transform_method)
    # prep.transform_data_pd( test_csv_path, test_hdf5_path, xtest, ytest, method = transform_method)
    
    ##Use the training data to fit the weights of the model
    nnm.train(  xtrain,
                ytrain,
                xval,
                yval,
                shape,
                activation_function,
                l2_reg,
                epochs,
                batch_size,
                learning_rate,
                train_analysis,
                val_analysis,
                model_dir,
                'testing')
    
    ##Use the saved h5 file to predict with the model
    nnm.test(xtest,
              ytest, 
              test_analysis,
              model_dir)
    
    # postp.error_analysis(predictions, test_labels, error_stats)
    
    
    # pt.plot_test(filtered_csv,filtered_hdf5, True)
    # pt.earthquake_test(filtered_csv,filtered_hdf5)
    # pt.noise_test(filtered_csv,filtered_hdf5)
    
if __name__ == '__main__':
    main()
        
