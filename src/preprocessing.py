# -*- coding: utf-8 -*-
import h5py
import sys
sys.path.insert(1, './../../src/')
from util import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
# import python_speech_features
import os
from datetime import datetime

@debugprint
def filter_merge_dataset(noise_paths, earthquake_paths, N_eq, N_noise, output_filtered_csv_path, output_filtered_hdf5_path):
    '''
    Filter out N-eq percent of the earthquake data / N-noise percent of the noise data
    '''
    #sample a fraction from the entire dataset
    df_filtered = pd.DataFrame()
    
    with h5py.File(output_filtered_hdf5_path, 'w') as hf_filtered:
        G0_filtered = hf_filtered.create_group('data')
        
        print(f'Processing Noise Files')
        for noise_files in noise_paths:
            df_temp = pd.read_csv(noise_files[0])
            df_temp = df_temp.sample(frac = N_noise)
            df_filtered = df_filtered.append(df_temp)
            filtered_traces = df_temp['trace_name']
            
            with h5py.File(noise_files[1], 'r') as hf_tmp:
                for trace_name in filtered_traces:
                    dataset = hf_tmp.get('data/' + trace_name)
                    G0_filtered.create_dataset(trace_name, data= dataset )
                    G0_filtered[trace_name].attrs.update(dataset.attrs)
            
        print(f'Processing Earthquake Files')    
        for eq_files in earthquake_paths:
            df_temp = pd.read_csv(eq_files[0])
            df_temp = df_temp.sample(frac = N_eq)
            df_filtered = df_filtered.append(df_temp)
            filtered_traces = df_temp['trace_name']
            
            with h5py.File(eq_files[1], 'r') as hf_tmp:
                for trace_name in filtered_traces:
                    dataset = hf_tmp.get('data/' + trace_name)
                    G0_filtered.create_dataset(trace_name, data=dataset)
                    G0_filtered[trace_name].attrs.update(dataset.attrs)
            
    df_filtered.to_csv(output_filtered_csv_path, index = False)

@debugprint
def split(filtered_csv_path, filtered_hdf5_path, K_split, 
          train_csv_path, train_hdf5_path, 
          val_csv_path, val_hdf5_path,
          test_csv_path, test_hdf5_path):
    
    #Check if the split sum equals to 1
    assert sum(K_split) == 1
    
    #Read the filtered file that contains all of our data
    df_filtered = pd.read_csv(filtered_csv_path)
    
    #Split into train/val/test
    df_train, df_val, df_test = np.split(df_filtered.sample(frac=1, random_state=42), 
                             [int(K_split[0]*len(df_filtered)), int( (K_split[0] + K_split[1])*len(df_filtered))])
    
    train_traces = df_train['trace_name']
    val_traces = df_val['trace_name']
    test_traces = df_test['trace_name']
    
    #Populate the train hdf5
    with h5py.File( train_hdf5_path, 'w') as hf_train:
        G0_filtered = hf_train.create_group('data')
        with h5py.File( filtered_hdf5_path, 'r') as filtered_hf:
            for trace_name in train_traces:
                dataset = filtered_hf.get('data/' + trace_name)
                G0_filtered.create_dataset(trace_name, data= dataset )
                G0_filtered[trace_name].attrs.update(dataset.attrs)
    print('Finished Train set Split')    
        
    #Populate the val hdf5
    with h5py.File(val_hdf5_path, 'w') as hf_val:
        G0_filtered = hf_val.create_group('data')
        with h5py.File( filtered_hdf5_path, 'r') as filtered_hf:
            for trace_name in val_traces:
                dataset = filtered_hf.get('data/' + trace_name)
                G0_filtered.create_dataset(trace_name, data= dataset )
                G0_filtered[trace_name].attrs.update(dataset.attrs)
    print('Finished Val set Split') 
    
    #Populate the test hdf5
    with h5py.File(test_hdf5_path, 'w') as hf_test:
        G0_filtered = hf_test.create_group('data')
        with h5py.File( filtered_hdf5_path, 'r') as filtered_hf:
            for trace_name in test_traces:
                dataset = filtered_hf.get('data/' + trace_name)
                G0_filtered.create_dataset(trace_name, data= dataset )
                G0_filtered[trace_name].attrs.update(dataset.attrs)
    print('Finished Test set Split') 
    
    df_train.to_csv(train_csv_path, index = False)
    df_val.to_csv(val_csv_path, index = False)
    df_test.to_csv(test_csv_path, index = False)
    
    #Print out stats of the data set:
    eq_in_train = len(df_train[ df_train["trace_category"] =="earthquake_local"])
    noise_in_train = len(df_train[ df_train["trace_category"] =="noise"])    
    
    eq_in_val = len(df_val[ df_val["trace_category"] =="earthquake_local"])
    noise_in_val = len(df_val[ df_val["trace_category"] =="noise"])
    
    eq_in_test = len(df_test[ df_test["trace_category"] =="earthquake_local"])
    noise_in_test = len(df_test[ df_test["trace_category"] =="noise"])     
        
    
    print(f'No. of total examples in train set: {len(df_train)}')
    print(f'No. of EQ in train set: {eq_in_train}')
    print(f'No. of Noise in train set: {noise_in_train}')
    
    print(f'No. of total examples in test set: {len(df_test)}')
    print(f'No. of EQ in test set: {eq_in_test}')
    print(f'No. of Noise in test set: {noise_in_test}')
    
    print(f'No. of total examples in val set: {len(df_val)}')
    print(f'No. of EQ in val set: {eq_in_val}')
    print(f'No. of Noise in val set: {noise_in_val}')
    
@debugprint    
def transform_data_pd(csv_file_path, hdf5_file_path , data_path, labels_path, method = None):
    '''
    Generate Training and testing data and labels, this should occur right before feeding the model
    Continuously output to csv files as the data gets transformed so rate doesnt slow down
    '''
    
    try:
        os.remove(data_path)
        os.remove(labels_path)
    except OSError:
        pass
    
    supported_methods = ['mean', 'three_channel_mean', 'mfcc', 'identity']
    if method in supported_methods: 
        print(f'Data transformation selected: {method} ')
    else:
        assert False, "Mode not supported"
    
    df_metadata = pd.read_csv(csv_file_path)
    df_metadata = df_metadata.sample(frac = 1)
    trace_list = df_metadata['trace_name'].to_list()

    with h5py.File(hdf5_file_path, 'r') as h5_data:
        data_arr = np.array([])
        label_arr = np.array([])
        
        for i, trace in enumerate(trace_list):
            dataset = h5_data.get('data/' + str(trace))
            data = np.array(dataset)
            if method == 'mean':
                transformed_data = mean_transform(data)
            elif method == 'three_channel_mean':
                transformed_data = three_channel_mean_transform(data)
            elif method == 'mfcc':
                transformed_data = mfcc(data)
            elif method == 'identity':
                transformed_data = identity(data)
            data_label = int(df_metadata[df_metadata['trace_name'] == trace]['trace_category'].iloc[0] == 'earthquake_local')
            if (i % 1000) == 0: print(i, datetime.now().strftime("%H:%M:%S"))
            # print(i)
            pd.DataFrame(data = [transformed_data]).to_csv(data_path, mode = 'a', header = False, index = False)
            pd.DataFrame( data = np.array([[data_label]])).to_csv(labels_path, mode = 'a', header = False, index = False)
            
    #Print relevant information
    print(f'shape of data array: {i, transformed_data.shape[0]}')
    
def mean_transform(arr):
    return np.mean(arr, axis=0)

def three_channel_mean_transform(arr):
    return block_reduce(arr, block_size=(2,3), func=np.mean, cval=np.mean(arr))[:,0]

def mfcc(arr):
    transform = python_speech_features.mfcc(arr, samplerate=100, winlen = 5, winstep = 5, numcep=10,nfilt=10,nfft=512)
    return transform.flatten()

def identity(arr): #return (18000,)
    out = np.concatenate((
            block_reduce(arr.T[0], block_size=(10,), func=np.mean),
            block_reduce(arr.T[1], block_size=(10,), func=np.mean),
            block_reduce(arr.T[2], block_size=(10,), func=np.mean)
            ))
    return out
    
    
    


