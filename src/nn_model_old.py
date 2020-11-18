# -*- coding: utf-8 -*-
import h5py
import sys
sys.path.insert(1, './../../src/')
from util import *
import os
import shutil

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import islice
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import model_from_json

@debugprint
def train(xtrainp, ytrainp, xvalp, yvalp, 
          shape = [], af = 'relu', l2_reg = 0, epochs = 150, bs = 10, lr = .001, 
          train_analysis = 'train_analysis.csv', train_trace = '',
          val_analysis = 'val_analysis.csv', val_trace = '',
          model_dir = './model/', tag = ''):
    '''
    Train a model and save off model architecture and weights
    '''
    #if model path exists, delete it
    if os.path.exists(model_dir):
        shutil.rmtree( model_dir )
        os.mkdir( model_dir )
    else: #Else make the path
        os.mkdir( model_dir )
    
    #Preserve the arg list somewhere to note it down at the end of the function
    arg_list = [[k,v] for k,v in locals().items()]
    
    #Extract data from csvs
    xtrain = pd.read_csv(xtrainp, header = 0)
    ytrain = pd.read_csv(ytrainp, header = 0)
    xval = pd.read_csv(xvalp, header = 0)
    yval = pd.read_csv(yvalp, header = 0)
    
    #Numpy arrays needed for training/validation
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xval = np.array(xval)
    yval = np.array(yval)
    
    #Specify the input dimensions
    input_dim = xtrain.shape[1] 
    
    #Create Model
    model = tf.keras.models.Sequential()

    #First Layer
    model.add(tf.keras.Input(shape=(input_dim,)))

    #Hidden Layers
    for i in shape:
        model.add( tf.keras.layers.Dense(i, 
                                         activation=af, 
                                         kernel_regularizer=l2(l2_reg)))
    #Last Layer
    model.add( tf.keras.layers.Dense(1, 
                                     activation='sigmoid', 
                                     kernel_regularizer=l2(l2_reg)) )
    
    #Optimizer
    adam = tf.keras.optimizers.Adam(lr=lr)
    
    #Compile Model for binary classification
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    #Train the model
    history = model.fit(xtrain, ytrain, 
                        validation_data = (xval, yval) ,
                        epochs=epochs , batch_size=bs)
    
    #Plot train and val curves
    fig = plt.figure(figsize = (12,10))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'model accuracy: {tag}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    fig.savefig(model_dir + 'model_curve.png', dpi=fig.dpi)
    
    #Evaluate model on training data
    t_loss, t_accuracy = model.evaluate(xtrain, ytrain, verbose = 0)
    print(f'Training Accuracy: {t_accuracy*100}')
    
    v_loss, v_accuracy = model.evaluate(xval, yval, verbose = 0)
    print(f'Validation Accuracy: {v_accuracy*100}')
    
    #Collect the training labels, predictions, and correctness and output to train analysis
    predictions = model.predict(xtrain)
    correct = [ y_gt == round(y_pred[0])  for y_gt, y_pred in zip(ytrain, predictions) ]
    t_analysis =  [ [y_gt[0],round(y_pred[0]), int(c)]  for y_gt, y_pred, c in zip(ytrain, predictions, correct)]
    t_analysis = pd.DataFrame(t_analysis, columns = ['Ground_Truth', 'Predicted', 'Correct'])
    t_analysis['trace_name'] = pd.read_csv(train_trace)['trace_name']
    t_analysis.to_csv(model_dir + train_analysis, index = False )
    
    #Collect the training labels, predictions, and correctness and output to val analysis
    predictions = model.predict(xval)
    correct = [ y_gt == round(y_pred[0])  for y_gt, y_pred in zip(yval, predictions) ]
    v_analysis =  [ [y_gt[0],round(y_pred[0]), int(c)]  for y_gt, y_pred, c in zip(yval, predictions, correct)]
    v_analysis = pd.DataFrame(v_analysis, columns = ['Ground_Truth', 'Predicted', 'Correct'])
    v_analysis['trace_name'] = pd.read_csv(val_trace)['trace_name']
    v_analysis.to_csv(model_dir + val_analysis, index = False )

    #Output/Save away the Model Arch
    model_json = model.to_json()
    with open(model_dir + 'model.json', "w") as json_file:
        json_file.write(model_json)
    
    #Save Model weights
    model.save_weights(model_dir + 'model.h5')
    
    #Note down hyperparameters used in the training
    with open(model_dir + 'model_hp.txt', 'w') as hp:
        for p,a in arg_list:
            hp.writelines(f'{p} : {a}\n')
        hp.writelines(f'Training accuracy : {t_accuracy*100}\n')    
        hp.writelines(f'Validation accuracy : {v_accuracy*100}\n')
    print('Saved Model Weights/Architecture')
 
@debugprint
def test(xtest, ytest, 
         test_analysis = 'test_analysis.csv', test_trace = '',
          model_dir = './model/'):
    '''
    Use an existing Model architecture and weights to execute Inference
    '''
    
    #Extract data from csvs
    xtest = pd.read_csv(xtest)
    ytest = pd.read_csv(ytest)

    #Numpy arrays needed for testing
    xtest = np.array(xtest)
    ytest = np.array(ytest)
    
    #Load up model via json
    loaded_model = tf.keras.models.model_from_json(open(model_dir + 'model.json', 'r').read())
    
    #load weights into new model via h5
    loaded_model.load_weights(model_dir + 'model.h5')
    
    #Compile Model
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    loss, accuracy = loaded_model.evaluate(xtest, ytest, verbose = 0)
    print('Testing Accuracy: %.2f' % (accuracy*100))
    
    #Collect the testing labels, predictions, and correctness and output to train analysis
    predictions = loaded_model.predict(xtest)
    correct = [ y_gt == round(y_pred[0])  for y_gt, y_pred in zip(ytest, predictions) ]
    t_analysis =  [ [y_gt[0],round(y_pred[0]), int(c)]  for y_gt, y_pred, c in zip(ytest, predictions, correct)]
    t_analysis = pd.DataFrame(t_analysis, columns = ['Ground_Truth', 'Predicted', 'Correct'])
    t_analysis['trace_name'] = pd.read_csv(test_trace)['trace_name']
    t_analysis.to_csv(model_dir + test_analysis, index = False )
    
    




    
