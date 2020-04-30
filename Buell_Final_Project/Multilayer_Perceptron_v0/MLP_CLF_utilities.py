"""
Landon Buell
Marek Petrik
CS 750.01
18 April 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics

            #### ACTIVATION FUNCTIONS ####

def sigmoid(z):
    """ Logistic Sigmoid activation function """
    return 1 / 1 + np.exp(-z)

def sigmoid_deriv(z):
    """ 1st Derivative of Logisiting Sigmoid Activation Function """
    return z * (1 - z)

def ReLU (z):
    """ Rectified Linear Unit activation function """
    return np.max([0,z])

            #### MISC OPERATIONAL FUNCTIONS ####

def generate_random_array(arr_shape,scalar=1):
    """ Generate random array of given shape """
    return np.random.random(size=arr_shape)*scalar


def RSS_Loss (x,y):
    """ Compute & return MSE loss function value """
    x,y = x.ravel(),y.ravel()   # flatten
    return np.sum((x-y)**2)     # compute RSS


            #### CLASS OBJECT DEFINITIONS ####

class MLP_Classifier ():
    """
    Create Multilayer Perceptron Classifier Object
    --------------------------------
    name (str) : Name at attach to MLP for user use
    layer_sizes (iter) : Iterable of intergers. 
        I-th element is the number of neurons in the I-th hidden layer
    n_features (int) : Number of features for network
    n_targets (int) : Number of target classes
    batch_size (int) : Number of samples in a single mini-batch
    max_iters (int) : Indicate maximum number of training iters
    --------------------------------
    Return initialied instance
    """

    def __init__(self,name,layer_sizes,n_features,n_classes,
                 momentum,
                 batch_size=100,max_iters=100):
        """ Initialize Object Instance """
        self.name = name                    # name of network
        self.layer_sizes = layer_sizes      # sizes of hidden layers
        self.depth = len(layer_sizes)       # number of hidden layers
        self.n_features = n_features        # number of input features
        self.n_classes = n_classes          # number of output classes
        self.momentum = momentum            # scalar for SGD
        self.batch_size = batch_size        # sizes of mini-batches for training
        self.max_iters = max_iters          # maximum iterations over data
        # generate weighting matrices & bias vectors 
        self.weight_dims,self.bias_dims = \
                self.transformation_dims()      # dimensions for weight & bias arrays
        self.weights = self.weighting_matrices()# generate weighting matrices
        self.biases = self.bias_vectors()       # generate bias vectors
        self.losses = np.array([])              # hold loss functions

    def transformation_dims (self):
        """ Generate all dimesnions for all weighting & bias vectors """
        self.all_layers = np.insert(self.layer_sizes,[0,self.depth],
                               [self.n_features,self.n_classes])

        # arrays to hold dimesnions of arrays
        weight_dims = []
        bias_dims = []

        if self.layer_sizes == (0,):
            weight_dims.append((self.n_classes,self.n_features))
            bias_dims.append((self.n_classes))

        else:        
            # iterate over all layers & assign shapes
            for layer in range (0,len(self.all_layers)-1):
                weight_dims.append((self.all_layers[layer+1],
                                    self.all_layers[layer]))
                bias_dims.append((self.all_layers[layer+1]))

        return weight_dims,bias_dims

    def weighting_matrices (self):
        """ Generate weighting matricies to transform between layers """
        weights = []                            # list to hold weighting arrays
        for arr_shape in self.weight_dims:      # for each tuple
            rand_vals = generate_random_array(arr_shape)    # generate params of that state
            weights.append(rand_vals)                       # add np arr to list
        return weights                          # return the list

    def bias_vectors (self):
        """ Generate bias vectors to transform between layers """
        biases = []                             # list to hold bias vectors
        for arr_shape in self.bias_dims:        # for each tuple
            rand_vals = generate_random_array(arr_shape)    # generate params for that state
            biases.append(rand_vals)                        # add np arr to list
        return biases                           # return the list

    def target_matrix(self,y):
        """ Build target matrix from target vector """
        n_samples = len(y)
        Y = np.zeros(shape=(n_samples,self.n_classes))
        for I in range(0,n_samples):
            target_class = y[I]
            Y[I][target_class] += 1
        return Y

    def forward_pass (self,x):
        """ Pass single sample through Network to last layer """
        PRE_ACTS = []              # before act func is applied
        FIN_ACTS = []              # hold all activations by layer      
        for layer in range (0,len(self.all_layers)-1):
            # exact elements & reshape
            W = self.weights[layer]     # extract weight matrix
            b = self.biases[layer]      # extract bias vector
            # use FP equations
            z = W @ x + b       # pre-activation
            PRE_ACTS.append(z)  # add to vec to list              
            x = 1*z             # identity activation
            FIN_ACTS.append(x)  # add vec to list
        # finished with all layers
        self.pre_activations = PRE_ACTS
        self.activations = FIN_ACTS
        return self

    def predict(self,X):
        """ Make predcitions based on feature matrix X """
        Z = np.array([])        # array to hold precited vals
        for x in X:             # each row:
            # run forward pass on sample
            self.forward_pass(x.transpose())    # forward pass data
            output = self.activations[-1]       # output is last activations
            Z = np.append(Z,np.argmax(output))  # add prediction to array
        return Z                # return arr of predictions
        
    def train_model (self,X,y):
        """ train model given feature matrix X & targets y """
        N_samples = X.shape[0]          # number of samples
        Y = self.target_matrix(y)       # build target matrix
        batch_size = self.batch_size    # set batch size

        # iterate through data
        for I in range (self.max_iters):
            print("Iteration:",I)
            self.print_weights()
            shuffled = np.random.permutation(N_samples)
            X,Y = X[shuffled],Y[shuffled]       # shuffle data            
            X_batch,Y_batch = X[:batch_size],Y[:batch_size]

            weight_grads,bias_grads,batch_loss = \
                self.mini_batches(X_batch,Y_batch,batch_size)

            # update matrices based on gradients for mini-batch
            self.losses = np.append(self.losses,batch_loss)
            for layer in range(len(self.weights)):
                self.weights[layer] += weight_grads[layer]
                self.biases[layer] += bias_grads[layer]

        return self                             # return updated model

    def mini_batches (self,X_batch,Y_batch,batch_size):
        """ Iterate through samples in a mini-batch of data """

        # Hold chnages in weights & biases for full mini-batch
        minibatch_weight_grads = [np.zeros(W.shape) for W in self.weights]
        minibatch_bias_grads = [np.zeros(b.shape) for b in self.biases]

        # In mini-batch subset:
        batch_loss_avg = 0
        for x,y in zip(X_batch,Y_batch):        # one sample at a time
            self.forward_pass(x.transpose())    # forward pass
            output = self.activations[-1]       # network output
            batch_loss_avg += RSS_Loss(output,y)# compute sample loss
            # SGD ON SINGLE SAMPLE SGD(output,target)
            dW,db = self.back_propagate(output,y)   # compute gradients for single sample
            for layer in range(len(self.weights)):  # add changes to grads for single samples
                minibatch_weight_grads[layer] += dW[layer]
                minibatch_bias_grads[layer] += db[layer]
        # Now, we have gone through all samples in the mini-batch
        batch_loss_avg /= batch_size            # avg loss for this mini-batch
        for layer in range(len(self.weights)):  # average dw & dW over batch size 
                minibatch_weight_grads[layer] /= batch_size
                minibatch_bias_grads[layer] /= batch_size
        # return avergaged bias & weight grads & avg loss for batch
        return minibatch_weight_grads,minibatch_bias_grads,batch_loss_avg

    def back_propagate(self,x,y):
        """ Perform SGD Back-propagation given activations, weights & biases 
            This has been adapted from Goodfellow's Deep Learning, Algorithm 6.4 """
        # hold activations (initialize w/ 0's)
        weight_grads = [np.zeros(W.shape) for W in self.weights]
        bias_grads = [np.zeros(b.shape) for b in self.biases]
        dx = 2*(x - y).reshape(1,-1)                 # output - input
        for l in range (len(self.weights)-1,0,-1):
            # Change is biases , weights & next layer
            db = dx * self.pre_activations[l]       # change in baises
            dW = dx.reshape(-1,1) @ self.activations[l-1].reshape(1,-1)        # change in weights
            dx = self.weights[l].transpose() * db       # next layer

            weight_grads[l] += dW           # update weights
            bias_grads[l] += db.ravel()     # update biases
        return weight_grads,bias_grads

    def print_weights(self):
        """ print out all weighting matrices """
        for layer in self.weights:
            print(layer)
            

            #### METRIC FUNCTIONS ####

def confusion_matrix (model,y,z,show=False):
    """
    Generate Confusion Matrix for Specific Model
    --------------------------------
    model (class) : Instance of trained MLP model
    y (array) : true testing values (n_samples x 1)
    z (array) : predicted testing samples (n_samples x 1)
    show (bool) : If True, visualize color-coded confusion matrix
    --------------------------------
    Return model w/ confusion matrix (n_classes x n_classes) attrb
    """
    confmat = metrics.confusion_matrix(y,z)
    setattr(model,'confusion_matrix',confmat)
    if show == True:
        plt.title(str(model.name),size=20,weight='bold')
        plt.xlabel('Predicted Classes',size=16,weight='bold')
        plt.ylabel('Actual Classes',size=16,weight='bold')
        plt.imshow(confmat,cmap=plt.cm.binary)
        plt.show()
    return model

        
