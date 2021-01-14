# -*- coding: utf-8 -*-
"""q3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1b-cNj34oz6V5wlt_Q5uUgpy8VJUhcSFN
"""

# from google.colab import drive
# drive.mount('/content/drive')

# import os
# os.chdir("/content/drive/My Drive/ML-Sem5/ML_Assignment_3/")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import plotly.graph_objects as go
import matplotlib.pyplot as plt

import random
import time

df_train = pd.read_csv('./Data_Q3/largeTrain.csv', header=None)
df_test = pd.read_csv('./Data_Q3/largeValidation.csv', header=None)

class Dataset_Q3(Dataset):

    def __init__(self, df):

        self.labels = df[0].to_numpy()

        self.inp = df[list(range(1,129))].to_numpy()


    def __len__(self):
        return self.labels.shape[0]
    

    def __getitem__(self, idx):

        ip = torch.tensor(self.inp[idx,:], dtype=torch.float)
        

        label = torch.tensor(self.labels[idx]) 

        return {'X': ip, 'y': label}

train_dataset = Dataset_Q3(df_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = Dataset_Q3(df_test)
test_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

class Ques3(nn.Module):
    def __init__(self, ip_size, hidden_size, op_size):
        super(Ques3, self).__init__()
        self.fc1 = nn.Linear(ip_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, op_size)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

def train(model, loss_fn, optimizer, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, device='cpu'):
    """Train the BertClassifier model.
    """
    # Start training loop
    best_acc_val = 0
    print("Start training...\n")
    for epoch_i in range(epochs):
        ## Training 

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            X, y = batch["X"], batch["y"]

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(X)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, y)

            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        if ((epoch_i+1) % 20) == 0 :
            print("Epoch : {} | Training Loss : {}".format(epoch_i+1, avg_train_loss))
        # Evaluation
        if evaluation == True:
            # After an epoch - validation 
            val_loss, val_accuracy = evaluate(model, loss_fn, val_dataloader)     
            if ((epoch_i+1) % 20) == 0 :       
                print("Epoch : {} | Validation Loss : {} , Validation Accuracy : {}".format(epoch_i+1, val_loss, val_accuracy))
    
    print("Training complete!")

    return avg_train_loss, val_loss, val_accuracy
    
    
    
def evaluate(model, loss_fn, val_dataloader):
    ## put the model in evaluation mode 
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        X, y = batch["X"], batch["y"]

        # Compute logits
        with torch.no_grad():
            logits = model(X)

        # Compute loss
        loss = loss_fn(logits, y)

        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == y).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

class GridSearch():
    """
    My implentation of Grid Search

    """
    def __init__(self, model, name_param, parameters):

        self.model = model
        self.name_param = name_param
        self.parameters = parameters
        self.best_model = None
        self.max_acc = 0
        self.performance_dict_train = {}
        self.performance_dict_val = {}

    def fit(self, train_loader, test_loader, EPOCHS=10):  
        """
        Trains and validates model for all the specified parameters using k-fold 
        cross validation

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.

        k : number of folds.

        """
        print("The parameter value to be varied :", self.name_param)
        ## Loop over the valuse for the specified parameter 
        for idx, param_value in enumerate(self.parameters):
            print("The current value of the parameter :", param_value)


            

            if self.name_param == "hidden_units":
                curr_model = self.model(128, param_value, 10)
            else:
                curr_model = self.model(128, 4, 10)
            
            # Create the optimizer
            if self.name_param == "learning_rate":
                optimizer = torch.optim.SGD(curr_model.parameters(), lr=param_value)
            else:
                optimizer = torch.optim.SGD(curr_model.parameters(), lr=0.01)

            ## use k-fold cross validation to train and test the model for the given parameter 
            train_loss, val_loss, val_acc = train(curr_model, nn.CrossEntropyLoss(), optimizer, train_loader, test_loader, epochs=EPOCHS, evaluation=True)

            print("Train ", train_loss)
            print("val loss :", val_loss)
            ## save the performace of the model for the parameter value 
            self.performance_dict_train[param_value] = train_loss
            self.performance_dict_val[param_value] = val_loss


    def plot_loss_vs_param(self):
        """
        Plots the average training and validation accuracies for different parameter values 
        """
        plt.plot(list(self.performance_dict_train.keys()), list(self.performance_dict_train.values()), marker='o', linestyle='--', label="Training")
        plt.plot(list(self.performance_dict_val.keys()), list(self.performance_dict_val.values()), marker='o', linestyle=':', label="Validation")
        plt.legend()
        plt.title("Training and Validation Loss as a function of "+self.name_param)
        plt.savefig("./Plots/Q3_"+self.name_param+".png")
        plt.show()

gs = GridSearch(Ques3, "hidden_units", [5, 20, 50, 100, 200])

gs.fit(train_loader, test_loader, EPOCHS=100)

gs.plot_loss_vs_param()

gs = GridSearch(Ques3, "learning_rate", [0.1, 0.01, 0.001])

gs.fit(train_loader, test_loader, EPOCHS=200)

gs.plot_loss_vs_param()
