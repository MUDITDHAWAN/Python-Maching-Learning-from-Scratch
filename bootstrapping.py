# -*- coding: utf-8 -*-
"""ques2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iTzlzhf7TkUxIZ68w6VxnYaBTJaPaUir
"""

# from google.colab import drive
# drive.mount('/content/drive')

import os
# os.chdir("/content/drive/My Drive/ML-Sem5/ML_Assignment-2/Dataset/")
os.chdir("./Dataset/")
import pandas as pd
import h5py
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

## Read the data file 
df = pd.read_csv("weight-height.csv")

## Extract features to be used in the question  
X = df['Height'].to_numpy()
Y = df['Weight'].to_numpy()

## Shuffle the data
X, y = shuffle(X, Y, random_state=0)
X = X.reshape((X.shape[0], 1))
y = y.reshape((y.shape[0], 1))

## Keep the 'T' subset of data separate to be used afterwards for testing purposes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def create_bootstrap_sample(X,y, nb_samples_1_bootstrap, nb_bootstrap_samples):
    """
    A function that creates 'nb_bootstrap_samples' number of Bootstrap replicates  

    Parameters
    ----------

    X : 2-dimensional numpy array of shape (n_samples, n_features)
    y : 1-dimensional numpy array of shape (n_samples,)
    nb_samples_1_bootstrap : (Integer) number of samples in 1 bootstrap replicate 
    nb_bootstrap_samples : (Integer) number of bootstrap replicates 
    
    Yields
    -------
    dataX : 2-dimensional numpy array of shape (nb_samples_1_bootstrap, n_features)
    dataY : 1-dimensional numpy array of shape (nb_samples_1_bootstrap,)

    """

    # Pick random samples from the dataset to be included in the Bootstrap replicate 
    idx = np.random.choice(list(range(X.shape[0])), (nb_samples_1_bootstrap, nb_bootstrap_samples), replace=True)

    ## Number of features 
    nb_features = X.shape[1] 

    ## Loop over number of Bootstrap replicates to be generated (nb_bootstrap_samples)
    for i in range(idx.shape[1]):
        dataX = np.zeros((1,nb_features))
        dataY = np.zeros((1,))

        # For 1 bootstrap replicate pich random 'nb_samples_1_bootstrap' elements from the data 
        for j in range(idx.shape[0]):
            ## each of size 
            dataX = np.concatenate([dataX, X[idx[j][i],:].reshape((1,nb_features))], axis=0)
            dataY = np.concatenate([dataY, y[idx[j][i]].reshape((1,))], axis=0)

        ## Delete first zero-row which was added just for concatenation
        dataX = np.delete(dataX, (0), axis=0)
        dataY = np.delete(dataY, (0), axis=0)
        
        yield dataX, dataY

## Store prediction of the Trained model on the Test set (T) for each bootstrap replicate
y_preds = []

## Loop over the generated Bootstrp replicates and train a Linear Regression Model
for one_sample_X, one_sample_y  in create_bootstrap_sample(X_train,y_train, X_train.shape[0], 50):
    # print("Bootstrap Replicate ready")
    # print("Shape of 1 Bootstrap replicate (X, y) : ", one_sample_X.shape, one_sample_y.shape)

    ## Create a new instance of the model 
    clf = LinearRegression()
    clf.fit(one_sample_X, one_sample_y)

    ## Predict on the 'T' set 
    y_pred = clf.predict(X_test) 
    y_preds.append(y_pred)

    # print("MSE obtained on the test set (T) by using the model trained on 1 bootstrap replicate : ", mean_squared_error(y_test, y_pred))

def compute_avg_pred(y_preds):
    """
    A function that computes the average predictions from the Bootstrap replicates  

    Parameters
    ----------

    y_preds : list of length- nb_bootstrap_samples, containing 1-dimensional numpy arrays of shape (T.shape[0],)
    
    Returns
    -------
    avg_prediction : 1-dimensional array of average predictions of size  (T.shape[0],)

    """
    ## To store the average predicton 
    avg_pred = []
    nb_samples = len(y_preds)

    ## Loop over the number of samples in test set 
    for nb_pred in range(y_preds[0].shape[0]):
        sum_pred = sum([pred[nb_pred] for pred in y_preds])
        ## Calculate the average prediction for each datapoint in T set
        avg_pred.append(sum_pred / nb_samples)
    
    return np.asarray(avg_pred)

## Compute the average predictions 
avg_y_pred = compute_avg_pred(y_preds)

## Calculating Bias 
bias_sq_avg = ((avg_y_pred.reshape((avg_y_pred.shape[0], 1)) - y_test)**2).mean()


def compute_varaince(y_preds, avg_y_pred):
    """
    A function that computes the Variance of the Bootstrap replicates  

    Parameters
    ----------

    y_preds : list of length- nb_bootstrap_samples, containing 1-dimensional numpy arrays of shape (T.shape[0],)
    avg_y_pred : 1-dimensional array of size (T.shape[0],)
    
    Returns
    -------
    mean_variance : float - mean variance of the bootstrap samples 

    """
    ## number of bootstrap samples 
    B = len(y_preds)

    ## convert predictions into a 2-dimensional array 
    y_preds_arr = np.asarray(y_preds).T
    # print(y_preds_arr.shape, avg_y_pred.shape)

    avg_y_pred = avg_y_pred.reshape((avg_y_pred.shape[0], 1))

    ## Store the squared error from the average for each bootstrap replicate prediction
    sqd_error_from_avg = ((y_preds_arr - avg_y_pred)**2)

    variance = np.sum(sqd_error_from_avg, axis=1) / (B-1)

    mean_variance = variance.mean()
    
    return mean_variance

variance = compute_varaince(y_preds, avg_y_pred)

def compute_MSE(y_preds, y_true):
    """
    A function that computes the MSE of the Bootstrap replicates  

    Parameters
    ----------

    y_preds : list of length- nb_bootstrap_samples, containing 1-dimensional numpy arrays of shape (T.shape[0],)
    y_true : 1-dimensional array of size (T.shape[0],)
    
    Returns
    -------
    mean_mse : float - mean MSE of the bootstrap samples 

    """
    ## number of bootstrap samples 
    B = len(y_preds)

    ## convert predictions into a 2-dimensional array 
    y_preds_arr = np.asarray(y_preds).T
    # print(y_preds_arr.shape, avg_y_pred.shape)

    y_true = y_true.reshape((y_true.shape[0], 1))

    ## Store the squared error from the true values for each bootstrap replicate prediction
    sqd_error_from_true = ((y_preds_arr - y_true)**2)

    mse = np.sum(sqd_error_from_true, axis=1) / (B)

    mean_mse = mse.mean()
    
    return mean_mse

mse = compute_MSE(y_preds, y_test)
print("Avg MSE obtained on Test set (T) (averaged over number of bootstrap samples): ", mse)

print("MSE: {}, Variance average: {}, Bias (square) average: {}".format(mse, variance, bias_sq_avg))

print("Value : ",mse - (bias_sq_avg) - variance)