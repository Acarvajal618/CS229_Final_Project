# -*- coding: utf-8 -*-
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
from sklearn.linear_model import LogisticRegression

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# def gen_numpy(infile, N):
#     with open(infile,'rb') as infile:
#         while True:
#             for i in range(N):
#             arr = np.genfromtxt(gen, dtype=None)
#             if arr.shape[0]<N:
#                 return np.empty([0])
#             else:
            
#             yield arr

def entire_dataset(filepath, N= None):
    A = np.array([])
    with open(filepath,'rb') as f:
        try:
            while True:
                tmp = np.load(f).T
                if A.size == 0:
                    A = tmp
                else:
                    A = np.vstack([A, tmp])
        except:
            pass
    return A
               
def train(train_data ,train_labels, epsilon , step_size, max_iter, train_theta_output, mode = None):
    
    if mode == 'scikit_LogReg':
        xtrain = np.loadtxt(train_data) #entire_dataset(train_data)
        ytrain = np.loadtxt(train_labels) #entire_dataset(train_labels)
        print(f'Running regression')
        logisticRegr = LogisticRegression(solver = 'lbfgs', 
                                          fit_intercept = True,
                                          max_iter = 100000,
                                          verbose = 1,
                                          n_jobs = -1,
                                          tol = 1e-15)
        logisticRegr.fit(xtrain, ytrain)
        logisticRegr.coef_
        np.savetxt(train_theta_output, logisticRegr.coef_)
    
    if mode == 'GD':
        xtrain = pd.read_csv(train_data).to_numpy() #entire_dataset(train_data)
        ytrain = pd.read_csv(train_labels).to_numpy() #entire_dataset(train_labels)
        theta = np.zeros( shape = (1, xtrain.shape[1] ))
        for iteration in range(max_iter):
            
            hyp = sigmoid( -theta @ xtrain.T)
            jThetaGrad = ((ytrain.T - hyp) @ xtrain)
            jThetaHessian = xtrain.T @ (xtrain * (hyp * (1 - hyp)).T)
            theta_new = theta - step_size * (np.linalg.inv(jThetaHessian) @ jThetaGrad.T).T
            # print(f'\nIteration : {iteration}\nTheta new:\n{theta_new.T}\nTheta_current:\n{theta.T}')
            print(f'L1 Norm: {np.linalg.norm(theta_new - theta,1)}')
            
            if np.linalg.norm(theta_new - theta,1) < epsilon:
                print(f'Algorithm has converged in {iteration} iterations')
                theta = theta_new
                break
            theta = theta_new
        else:
            print(f'Algorithm has NOT convered in {iteration} iterations')
        pd.DataFrame(data = theta).to_csv(train_theta_output, mode = 'w', header = False, index = False)
        pd.DataFrame(data = theta).to_csv(train_theta_output, mode = 'a', header = False, index = False)
        print('done')
    # if mode == 'BGD':   
    #     if pargs: B = pargs[0]
        
        
        
    #         data = open(train_data_txt)
    #         labels = open(train_labels_txt)
        
        
    #             xtrain = np.array([])
    #             ytrain = np.array([])
                
    #             for iteration in range(max_iter):
    #                 for i in range(B):
    #                     try:
    #                         tmp = np.load(f).T
    #                         if xtrain.size == 0:
    #                             xtrain = np.load(f).T
    #                         else:
    #                             xtrain = np.vstack([xtrain , ])
         
                
    #             xtrain = load_batch(train_data_txt, B)
    #             ytrain = load_batch(train_labels_txt, B)
                
    #             for iteration in range(max_iter):
                    
    #                 try:
    #                     ytrain_batch = next(ytrain)
    #                     xtrain_batch = next(xtrain)
    #                 except:
    #                     xtrain = gen_numpy(train_data_txt, B)
    #                     ytrain = gen_numpy(train_labels_txt, B)
    #                     ytrain_batch = next(ytrain)
    #                     xtrain_batch = next(xtrain)
                        
    #                 if iteration == 0: theta = np.zeros( shape = ( 1, xtrain_batch.shape[1]))
                    
    #                 hyp = sigmoid( -theta @ xtrain_batch.T)
    #                 jThetaGrad = ((ytrain_batch - hyp) @ xtrain_batch).T/len(xtrain_batch)
    #                 # jThetaHessian = xtrain.T @ (xtrain * (hyp * (1 - hyp)).T)
    #                 # theta_new = theta - step_size * (np.linalg.inv(jThetaHessian) @ jThetaGrad).T
    #                 theta_new = theta - step_size * jThetaGrad.T
                    
    #                 # print(f'\nIteration : {iteration}\nTheta new:\n{theta_new.T}\nTheta_current:\n{theta.T}')
    #                 print(f'L1 Norm: {np.linalg.norm(theta_new - theta,1)}')
                    
    #                 if np.linalg.norm(theta_new - theta,1) < epsilon:
    #                     print(f'Algorithm has converged in {iteration} iterations')
    #                     theta = theta_new
    #                     break
    #                 theta = theta_new
                    
    #             else:
    #                 print(f'Algorithm has NOT convered in {iteration} iterations')
                    
    #             np.savetxt(train_theta_output, theta)
    
    # if mode == 'BGD':
    #     # if pargs: B = 100
        # data = open(train_data_txt)
        # labels = open(train_labels_txt)
            
        # for iteration in range(max_iter):
        #     xtrain = np.array([])
        #     ytrain = np.array([])
        #     for i in range(B):
                
                
        #         for iteration in range(max_iter):
        #             for i in range(B):
        #                 try:
        #                     tmp = np.load(f).T
        #                     if xtrain.size == 0:
        #                         xtrain = np.load(f).T
        #                     else:
        #                         xtrain = np.vstack([xtrain , ])
         
                
        #         xtrain = load_batch(train_data_txt, B)
        #         ytrain = load_batch(train_labels_txt, B)
                
        #         for iteration in range(max_iter):
                    
        #             try:
        #                 ytrain_batch = next(ytrain)
        #                 xtrain_batch = next(xtrain)
        #             except:
        #                 xtrain = gen_numpy(train_data_txt, B)
        #                 ytrain = gen_numpy(train_labels_txt, B)
        #                 ytrain_batch = next(ytrain)
        #                 xtrain_batch = next(xtrain)
                        
        #             if iteration == 0: theta = np.zeros( shape = ( 1, xtrain_batch.shape[1]))
                    
        #             hyp = sigmoid( -theta @ xtrain_batch.T)
        #             jThetaGrad = ((ytrain_batch - hyp) @ xtrain_batch).T/len(xtrain_batch)
        #             # jThetaHessian = xtrain.T @ (xtrain * (hyp * (1 - hyp)).T)
        #             # theta_new = theta - step_size * (np.linalg.inv(jThetaHessian) @ jThetaGrad).T
        #             theta_new = theta - step_size * jThetaGrad.T
                    
        #             # print(f'\nIteration : {iteration}\nTheta new:\n{theta_new.T}\nTheta_current:\n{theta.T}')
        #             print(f'L1 Norm: {np.linalg.norm(theta_new - theta,1)}')
                    
        #             if np.linalg.norm(theta_new - theta,1) < epsilon:
        #                 print(f'Algorithm has converged in {iteration} iterations')
        #                 theta = theta_new
        #                 break
        #             theta = theta_new
                    
        #         else:
        #             print(f'Algorithm has NOT convered in {iteration} iterations')
                    
        #         np.savetxt(train_theta_output, theta)    
        
def test(test_data_txt, theta_txt, predictions_txt):
    
    theta = pd.read_csv(theta_txt).to_numpy()
    xtest = pd.read_csv(test_data_txt).to_numpy()
    
    predictions = sigmoid( -theta @ xtest.T)
    
    np.savetxt(predictions_txt, predictions)
    
