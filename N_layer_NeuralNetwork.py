import numpy as np
import matplotlib.pyplot as plt
import pickle

def get_batch(X,y,batch_size=32):
  """
    Create k-folds for the given data set

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

    y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.

    batch_size : Specifies number of examples in one batch 

    Returns
    -------
    Generator : Batches of the given data  
    """
  ## X - (n_samples, n_features)

  nb_samples = X.shape[0]
  
  flag = 0
  nb_batches = (nb_samples // batch_size)
  batch_sizes = [batch_size] *nb_batches

  ## Number of batches id batch_size is no a divisor 
  if nb_samples % batch_size != 0:
    flag =1
    nb_batches += 1
    batch_sizes.append(nb_samples % batch_size)   
  
  start_idx =0

  for batch_idx, curr_size in enumerate(batch_sizes):
    # print("Batch Index: {}, Batch Size: {}".format(str(batch_idx+1), str(curr_size)))

    # print("batch idx", batch_idx)
    ## index for the end of one fold 
    end_idx = start_idx + curr_size

    # print("Start idx :", start_idx, "end idx :", end_idx)

    # print(X.shape, y.shape)
    ## returning a generator
    yield X[start_idx: end_idx, :].T, y[start_idx: end_idx,].reshape((curr_size,1)).T

    start_idx = end_idx # assigning end index to start index for the next fold
    


class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', "tanh", "softmax"]
    weight_inits = ['zero', 'random', 'normal']

    W = {}
    b = {}
    A_out = {}
    Z_out = {}
    dW = {}
    db = {}
    dA = {}

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')


        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs =num_epochs

        self.activation_output = 'softmax' ## Activation function for the output layer 

        ## Initialize the weights 
        self.init_params()

    def init_params(self):
        
        ## Choose the initialization function 
        if self.weight_init == 'zero':
            wt_init_func = self.zero_init

        elif self.weight_init == 'random':
            wt_init_func = self.random_init

        else:
            wt_init_func = self.normal_init

        ## Loop over the layers to initialize weights
        for layer in range(1, self.n_layers):
            self.W[str(layer)] = wt_init_func((self.layer_sizes[layer], self.layer_sizes[layer-1]))
            self.b[str(layer)] = np.zeros((self.layer_sizes[layer], 1))

            self.dW[str(layer)] = np.zeros((self.layer_sizes[layer], self.layer_sizes[layer-1]))
            self.db[str(layer)] = np.zeros((self.layer_sizes[layer], 1))

    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """

        x_calc = np.maximum(X, 0)

        return x_calc

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """

        x_calc = (X>0).astype(int)

        return x_calc

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        ## Avoid overflow
        X = np.clip(X, -300, 300)

        x_calc = 1/(1+np.exp(-X))

        return x_calc

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """

        A = self.sigmoid(X)
        
        x_calc = A * (1 - A)

        return x_calc

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """

        # x_calc = X

        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        x_calc = np.ones_like(X)

        return x_calc

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        ## Avoid overflow
        # X = np.clip(X, -100, 100)
        

        x_calc = (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

        return x_calc

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        
        A = self.tanh(X)
        
        x_calc = 1 - A**2

        return x_calc

    def softmax(self, X, axis=0):
        """
        Calculating the Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        X -= np.amax(X, axis=axis, keepdims=True) ## Normalize the data for numeric stability 
        x_calc = np.exp(X) / np.sum(np.exp(X), axis=axis, keepdims=True) ## Calculate the softmax output 

        return x_calc

    def softmax_grad(self, X, axis=0):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """

        soft_out = self.softmax(X, axis=axis)
        ## The jacobian matrix
        x_calc = - soft_out.reshape(soft_out.shape[axis], 1) * soft_out.reshape(soft_out.shape[axis], 1).T
        ## Subtract from the diagonal ( mimic the delta function)
        x_calc += np.diag(soft_out.flatten())

        return x_calc
    

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """
        
        weight = np.zeros(shape)

        return weight

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """

        weight = np.random.rand(shape[0], shape[1]) *0.01

        return weight

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 1-dimensional numpy array which contains the initial weights for the requested layer
        """

        weight = np.random.randn(shape[0], shape[1]) *0.01

        return weight
    
    def forward_linear_step(self, A, W, b):
        """
        Calculating thes dot product of weights and inputs for a particular layer

        Parameters
        ----------
        A : last layer's output
        
        W : weight matrix of the layer
        
        b : bias of the layer

        Returns
        -------
        Z : output of the linear transformation i.e., without non-linear activation the requested layer
        """
        Z = np.dot(W, A) + b ## carry out the matrix multiplication step (linear)

        return Z

    def forward_linear_activation_step(self, A_prev, W, b, activation="relu"):
        """
        Calculates the outptut of the for a particular layer with activation 

        Parameters
        ----------
        A_prev : last layer's output
        
        W : weight matrix of the layer
        
        b : bias of the layer
        
        activation : activation for the particular layer 

        Returns
        -------
        A : ouput with activation function 
        
        Z: output without activation function 
        """
        Z = self.forward_linear_step(A_prev, W, b) ## Carry out the linear step 

        ## Add the activation function on it 
        if activation == 'sigmoid':
            A = self.sigmoid(Z)
        elif activation == 'relu':
            A = self.relu(Z)
        elif activation == 'linear':
            A = self.linear(Z) 
        elif activation == 'tanh':
            A = self.tanh(Z) 
        else:
            A = self.softmax(Z, axis=0)

        return A, Z

    def forward_prop(self, X):
        """
        Performs forward propogation through the network 

        Parameters
        ----------
        X : input to the model  

        Returns
        -------
        A_ouput_layer : output of the model 
        """

        self.A_out['0'] = X ## Saving it for use in back prop

        # print("Shape of A[0] or input ", self.A_out['0'].shape)

        for layer in range(1,self.n_layers-1):
            # print("layer :", layer)

            # print("Shape of previous layer A :", self.A_out[str(layer-1)].shape)
            # print("Shape of curr W, b :", self.W[str(layer)].shape, self.b[str(layer)].shape)

            ## Calculate the output for one layer 
            A, Z = self.forward_linear_activation_step(self.A_out[str(layer-1)], self.W[str(layer)], self.b[str(layer)], self.activation)

            self.A_out[str(layer)] = A ## store for calculating gradient 
            self.Z_out[str(layer)] = Z ## store for calculating gradient 

            # print("Shape of out A :", self.A_out[str(layer)].shape)
            # print("Shape of curr Z out :", self.Z_out[str(layer)].shape)

        
        # print("Layer :", self.n_layers-1)
        # print("Shape of previous layer A :", A.shape)
        # print("Shape of curr W, b :", self.W[str(self.n_layers-1)].shape, self.b[str(self.n_layers-1)].shape)

        ## Calculate the ouput layer 
        A_ouput_layer, Z_ouput_layer = self.forward_linear_activation_step(A, self.W[str(self.n_layers-1)], 
                                                                           self.b[str(self.n_layers-1)], self.activation_output)

        self.A_out[str(self.n_layers-1)] = A_ouput_layer ## store for calculating gradient 
        self.Z_out[str(self.n_layers-1)] = Z_ouput_layer ## store for calculating gradient 

        # print("Shape of out A :", self.A_out[str(self.n_layers-1)].shape)
        # print("Shape of curr Z out :", self.Z_out[str(self.n_layers-1)].shape)


        return A_ouput_layer

    def cost_function(self, A_last, Y):
        """
        Calculates the cross entopy loss 

        Parameters
        ----------
        A_last : output of the model  
        
        Y : labels of the samples 

        Returns
        -------
        ce_loss : cross entropy loss 
        """
        ## Cross-Entropy Loss

        nb_samples = A_last.shape[1]

        ## Restrict the softmax output to a range [1e-12, 1-1e-12] for numeric stability 
        logits = np.clip(A_last, 1e-12, 1. - 1e-12)

        ## Convert labels into one-hot encoded vectors 
        enc_Y = np.zeros((Y.size, 10))
        enc_Y[np.arange(Y.size),Y] = 1 

        ## Calculate the log term 
        log_term = enc_Y.T * np.log(A_last) 

        # print("log term :", log_term)

        ## Calculate the cross-entropy loss 
        ce_loss = - np.sum(log_term) / nb_samples

        # print("C E loss :", ce_loss)
        return ce_loss
    

    def backward_linear_step(self, dZ, A_prev, W, b):
        """
        Calculates one step of the chain rule and calculates the gradients wrt to W and b and dA for prev layer 

        Parameters
        ----------
        dZ : gradient wrt to Z ( without activation) output of the particular layer 
        
        A_prev : output of last layer 
        
        W : weight matrix of the particular layer 
        
        b : bias of the layer
        
        activation : activation for the particular layer 

        Returns
        -------
        dA_prev : gradient wrt the output of previous layer -- to contine the chain rule/ back propogation 
        
        dW : gradient wrt W
        
        db : gradient wrt b
        """
                
        m = A_prev.shape[1]

        ## Claculate the gradient for the linear step
    
        dA_prev = np.dot(W.T,dZ) ## used for calculating gradient of the previosu layer

        dW = 1/m*(np.dot(dZ,A_prev.T)) ## gradient wrt the w of a layer 

        db = 1/m*(np.sum(dZ,axis=1,keepdims=True)) ## gradient wrt the b of a layer 

        

        return dA_prev, dW, db

    def backward_linear_activation_step(self, dA, A_prev, W, b, Z, activation):
        """
        Performs back progation through 1 layer  

        Parameters
        ----------
        dA : gradient wrt to A ( with activation) output of the particular layer 
        
        A_prev : output of last layer 
        
        W : weight matrix of the particular layer 
        
        b : bias of the layer
        
        Z : output of the particular layer without the activation function 
        
        activation : activation for the particular layer 

        Returns
        -------
        dA_prev : gradient wrt the output of previous layer -- to contine the chain rule/ back propogation 
        
        dW : gradient wrt W
        
        db : gradient wrt b
        """
        
        ## Choose the activation function 
        if activation == 'sigmoid':
            activation_gradient = self.sigmoid_grad(Z)
        elif activation == 'relu':
            activation_gradient = self.relu_grad(Z)
        elif activation == 'linear':
            activation_gradient = self.linear_grad(Z) 
        elif activation == 'tanh':
            activation_gradient = self.tanh_grad(Z) 
        else:
            activation_gradient = self.softmax_grad(Z)

        # print("activation_gradient", activation_gradient.shape)
        # print("dA", dA.shape)
        
        dZ = dA * activation_gradient ## dA -- gradient of the l+1 layer post-activation gradient

        dA_prev, dW, db = self.backward_linear_step(dZ, A_prev, W, b)

        return dA_prev, dW, db
    
    def ce_soft_grad(self, y, A_prev, W, b, Z):
        """
        Calculates gradient for softmax and cross entropy loss together and the gradients wrt the weights and bias for the last layer 

        Parameters
        ----------
        y : true labels  
        
        A_prev : output of last layer 
        
        W : weight matrix of the particular layer 
        
        b : bias of the layer
        
        activation : activation for the particular layer 

        Returns
        -------
        dA_prev : gradient wrt the output of previous layer -- to contine the chain rule/ back propogation 
        
        dW : gradient wrt W
        
        db : gradient wrt b
        """

        m = y.shape[1]
        grad = self.softmax(Z, axis=0)

        grad[y, range(m)] -= 1 

        dZ = grad/m

        dA_prev, dW, db = self.backward_linear_step(dZ, A_prev, W, b)

        return dA_prev, dW, db

    def backward_prop(self, AL, Y):
        """
        Performs the back propogation pass through the network  

        Parameters
        ----------
        AL : output of the last layer or the ouput for the forward propogation  
        
        Y : true labels 
        """

        L = self.n_layers - 1 ## Store the number of non-output layers 

        # print('Taking out dW and db for layer - ', L)

        ## Calculate gradient for the output layer 
        self.dA[str(L-1)], self.dW[str(L)], self.db[str(L)] =  self.ce_soft_grad(Y, self.A_out[str(L-1)], 
                                                                                self.W[str(L)], self.b[str(L)], 
                                                                                self.Z_out[str(L)])
        
        # print("Shape of dW", self.dW[str(L)].shape, " , db :", self.db[str(L)].shape, " , for layer : ", L )
        # print("dAshape :", self.dA[str(L-1)].shape, " for layer ", L-1)

        ## Loop over the hidden units 
        for l in reversed(range(L-1)):
            # print('Taking out dW and db for layer - ', l+1)

            self.dA[str(l)], self.dW[str(l+1)], self.db[str(l+1)] = self.backward_linear_activation_step(self.dA[str(l+1)], self.A_out[str(l)], 
                                                                                                    self.W[str(l+1)], self.b[str(l+1)], 
                                                                                                    self.Z_out[str(l+1)], self.activation)
            

            # print("Shape of dW", self.dW[str(l+1)].shape, " , db :", self.db[str(l+1)].shape, " , for layer : ", l+1 )
            # print("dAshape :", self.dA[str(l)].shape, " for layer ", l)

        return None



    def update_weights(self):
        """
        Updates the weights of the network with the gradients calculated during back prop   

        """
        
        # Update the parameters with gradient clipping 
        for l in range(1, self.n_layers ):

            self.W[str(l)] = self.W[str(l)] - (self.learning_rate * np.clip(self.dW[str(l)],-3, 3))
            self.b[str(l)] = self.b[str(l)] - (self.learning_rate * np.clip(self.db[str(l)],-3, 3))

        return None


    def fit(self, X, y, validation_data=None):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """

        np.random.seed(32)

        ## Keep track of training and validation loss
        costs = []   
        costs_test = []                      
        
        #Loop over number of epochs 
        for i in range(0, self.num_epochs):
            # print("X, y", X.shape, y.shape)

            cost_iter = [] ## Save cost for each iteration 
            ## Loop over number of batches 
            for X_batch, y_batch in get_batch(X,y,self.batch_size):
                # print("X, y", X_batch.shape, y_batch.shape)

                ## Forward propagation
                AL = self.forward_prop(X_batch)


                ## Calculate the loss 
                cost = self.cost_function(AL, y_batch)
                cost_iter.append(cost)


                ## Backward propagation
                self.backward_prop(AL, y_batch)

        
                # Update the parameters parameters.
                self.update_weights()
                
                # break

            costs.append(sum(cost_iter)/len(cost_iter)) ## save the training loss for the epoch

            # break

            ## Validation 
            if validation_data != None:
                # print("start valid")
                X_test, y_test = validation_data[0], validation_data[1]

                costs_test_iter = []
                for X_batch, y_batch in get_batch(X_test, y_test,self.batch_size):
                    # print("X, y", X_batch.shape, y_batch.shape)

                    ## Forward propagation
                    AL = self.forward_prop(X_batch)

                    ## Calculate the loss 
                    cost = self.cost_function(AL, y_batch)
                    costs_test_iter.append(cost)

                costs_test.append(sum(costs_test_iter)/len(costs_test_iter)) ## save the validation loss for the epoch

                if (i+1)%20 ==0:
                    print("Epoch %i | Training loss:  %f | Validation Loss: %f | Validation Accuracy: %f" %(i+1, costs[-1], costs_test[-1], self.score(X_test, y_test)))
            
            # break
                
        # plot the cost
        plt.plot(costs, label="Training Loss")
        plt.plot(costs_test, label="Validation Loss")
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title("Activation Function: " + self.activation +" | Learning rate =" + str(self.learning_rate))
        plt.show()

        return self

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """

        # return the numpy array y which contains the predicted values

        ## Pass through the network once 
        out_prob = self.forward_prop(X.T).T

        return out_prob

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        
        out_prob = self.forward_prop(X.T) 
        # print("Probability : ", out_prob.shape)

        ## Fnd max index for each sample 
        preds = np.argmax(out_prob, axis=0)

        # return the numpy array y which contains the predicted values
        return preds

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """

        y_pred = self.predict(X)

        # print("Y pred shape :", y_pred.shape)
        # print("Y shape:", y.shape)

        ##Find Accuracy
        accuracy = ((y_pred == y)*1.0).mean() * 100

        # return the numpy array y which contains the predicted values
        return accuracy
    
    def save_weights(self, file_name):
        ## Create a dictionary 
        model_dict = {'W': self.W, 'b': self.b, 'act_fn': self.activation}
        
        ## save the weights
        with open(file_name, 'wb') as f:
            pickle.dump(model_dict, f)
    
    def load_weights(self, file_name):
        
        ## read the weight file
        with open(file_name, 'rb') as f:
            model_dict = pickle.load(f)
        
        ## assign the weights 
        self.W, self.b, self.activation = model_dict['W'], model_dict['b'], model_dict['act_fn']