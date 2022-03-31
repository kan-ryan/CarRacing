import pickle
import numpy as np
import os
import gzip
import torch


def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")

    #combines the paths
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # crop, 84 x 84, some pixels never change
    X_train = np.array([img[:-12,6:-6] for img in X_train])
    X_valid = np.array([img[:-12,6:-6] for img in X_valid])
    # grayscale
    X_train = np.array([np.dot(img[...,0:3], [0.299, 0.587, 0.114]) for img in X_train])
    X_valid = np.array([np.dot(img[...,0:3], [0.299, 0.587, 0.114]) for img in X_valid])
    # scaling/normalizing
    X_train = np.array([img/255.0 for img in X_train])
    X_valid = np.array([img/255.0 for img in X_valid])
    
    return X_train, y_train, X_valid, y_valid
