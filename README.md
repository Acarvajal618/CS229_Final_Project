# CS229_Final_Project
CS229 Final project: Seismic Signal Detector


DIRECTORY STRUCTURE:  
Project  
|  
|-> analysis  
|  |  
|  |-> {Exp_Name}  
|    |  
|    |-> main.py (experiment file)  
|      |  
|      |-> model  
|      |  |->model_hp.txt (contains args used for training; HP)  
|      |  |->model.h5 (contains saved model weights)  
|      |  |->model.json (contain saved model architecture)  
|      |  
|      |-> results  
|        |-> test_analysis (test set labels/ our predictions/ if we were correct)  
|        |-> train_analysis (train set labels/ our predictions/ if we were correct)  
|        |-> val_analysis (val set labels/ our predictions/ if we were correct)  
|-> data  
|  |  
|  |-> {dataset_Name}  
|    |  
|    |-> Filtered.csv/.hdf5 + train/val/test csv/hdf5  
|  
|-> src (all the source code)   
|  
|-> original main  
    |  
    |-> main.py (baseline experiment file)  

The idea is we'll be running experiments in individual experiment folders in the analysis folder in order to preserve repro-ability
So there is a baseline experiment folder (T0_base) that we copy from to make new experiment folders.
From the experiment folder we'll run loops and sub experiments to test HP and tuning.
The main files in each experiment folder will pull from the data directory for train/test/val data.
All the functions called by main are in the src directory.
The 'main' files in each experiment will be different per experiment but if we wanted a baseline 'main.py'
then there's one saved in the original main directory.

The hdf5 files would all be in the data directory, theyre not included in the repo due to size

