# -*- coding: utf-8 -*-
import h5py
import sys
sys.path.insert(1, './../../src/')
from util import *

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
          train_analysis = './../train_analysis.csv',val_analysis = './../val_analysis.csv',
          model_arch = './model/model.json', model_weights = './model/model.h5',
          model_hp = './model/model_hp.txt'):
    '''
    Train a model and save off model architecture and weights
    '''
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
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    #Evaluate model on training data
    t_loss, t_accuracy = model.evaluate(xtrain, ytrain, verbose = 0)
    print(f'Training Accuracy: {t_accuracy*100}')
    
    v_loss, v_accuracy = model.evaluate(xval, yval, verbose = 0)
    print(f'Validation Accuracy: {v_accuracy*100}')
    
    #Collect the training labels, predictions, and correctness and output to train analysis
    predictions = model.predict(xtrain)
    correct = [ y_gt == round(y_pred[0])  for y_gt, y_pred in zip(ytrain, predictions) ]
    t_analysys =  [ [y_gt[0],round(y_pred[0]), int(c)]  for y_gt, y_pred, c in zip(ytrain, predictions, correct)]
    pd.DataFrame(t_analysys, columns = ['Ground_Truth', 'Predicted', 'Correct']).to_csv(train_analysis, index = False )
    
    #Collect the training labels, predictions, and correctness and output to val analysis
    predictions = model.predict(xval)
    correct = [ y_gt == round(y_pred[0])  for y_gt, y_pred in zip(yval, predictions) ]
    v_analysys =  [ [y_gt[0],round(y_pred[0]), int(c)]  for y_gt, y_pred, c in zip(yval, predictions, correct)]
    pd.DataFrame(v_analysys, columns = ['Ground_Truth', 'Predicted', 'Correct']).to_csv(val_analysis, index = False )

    #Output/Save away the Model Arch
    model_json = model.to_json()
    with open(model_arch, "w") as json_file:
        json_file.write(model_json)
    
    #Save Model weights
    model.save_weights(model_weights)
    
    #Note down hyperparameters used in the training
    with open(model_hp, 'w') as hp:
        for p,a in arg_list:
            hp.writelines(f'{p} : {a}\n')
            
    print('Saved Model Weights/Architecture')
 
@debugprint
def test(xtest, ytest, test_analysis = './../train_analysis.csv',
          model_arch = './model/model.json', model_weights = './model/model.h5'):
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
    loaded_model = tf.keras.models.model_from_json(open(model_arch, 'r').read())
    
    #load weights into new model via h5
    loaded_model.load_weights(model_weights)
    
    #Compile Model
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    loss, accuracy = loaded_model.evaluate(xtest, ytest, verbose = 0)
    print('Testing Accuracy: %.2f' % (accuracy*100))
    
    #Collect the testing labels, predictions, and correctness and output to train analysis
    predictions = loaded_model.predict(xtest)
    correct = [ y_gt == round(y_pred[0])  for y_gt, y_pred in zip(ytest, predictions) ]
    t_analysis =  [ [y_gt[0],round(y_pred[0]), int(c)]  for y_gt, y_pred, c in zip(ytest, predictions, correct)]
    pd.DataFrame(t_analysis, columns = ['Ground_Truth', 'Predicted', 'Correct']).to_csv(test_analysis, index = False )
    
    
    




    