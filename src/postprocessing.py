# -*- coding: utf-8 -*-
import h5py
import sys
sys.path.insert(1, './../../src/')
from util import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@debugprint
def error_analysis(predictions, test_labels_path, stats_file):
    pred = np.loadtxt(predictions)
    test_labels = pd.read_csv(test_labels_path).to_numpy()
    
    acc_arr = np.array([])
    for p, gt in zip(pred, test_labels):
        if (p >= .5 and gt == 1) or (p < .5 and gt == 0):
            acc_arr = np.append(acc_arr, 1)
        else:
            acc_arr = np.append(acc_arr, 0)
    
    model_accuracy = np.sum(acc_arr)/len(acc_arr)
    no_test_eqs = np.sum(test_labels)
    no_test_noise = len(test_labels) - np.sum(test_labels)
    total = len(test_labels)
    print(f'Amount of data in the testing set: {total}')
    print(f'Earthquakes in the testing set: {no_test_eqs}, {no_test_eqs/total}')
    print(f'Noise in the testing set: {no_test_noise}, {no_test_noise/total}')
    print(f'Accuracy of the model on the testing data is : {model_accuracy}')
