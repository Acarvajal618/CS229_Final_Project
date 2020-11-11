# CS229_Final_Project
CS229 Final project: Seismic Signal Detector

The following is the directory structure:

Project
|
|-> analysis
|   |
|   |-> {Exp_Name}
|       |
|       |-> main.py (experiment file)
|       |
|       |-> model
|        |   |->model_hp.txt (contains args used for training; HP)
|        |   |->model.h5 (contains saved model weights)
|        |   |->model.json (contain saved model architecture)
|        |
|        |-> results
|            |-> test_analysis (test set labels/ our predictions/ if we were correct)
|            |-> train_analysis (train set labels/ our predictions/ if we were correct)
|            |-> val_analysis (val set labels/ our predictions/ if we were correct)
|
|-> data
|   |
|   |-> {dataset_Name}
|       |
|       |-> Filtered.csv/.hdf5 + train/val/test csv/hdf5
|
|-> src (all the source code)   
|
|-> original main
    |
    |-> main.py (baseline experiment file)
