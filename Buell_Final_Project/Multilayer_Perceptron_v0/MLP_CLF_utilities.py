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
    --------------------------------
    Return initialied instance
    """

    def __init__(self,name,layer_sizes,n_features,n_classes):
        """ Initialize Object Instance """
        self.name = name                    # name of network
        self.layer_sizes = layer_sizes      # sizes of hidden layers
        self.depth = len(layer_sizes)       # number of hidden layers
        self.n_features = n_features        # number of input features
        self.n_classes = n_classes          # number of output classees
        # generate weighting matrices & bias vectors 
        self.weight_dims,self.bais_dims = \
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
        for arr_shape in self.bais_dims:        # for each tuple
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
        acts = [x]                      # hold all activations by layer
        for layer in range (0,len(self.all_layers)-1):
            # exact elements & reshape
            W = self.weights[layer]     # extract weight matrix
            b = self.biases[layer]      # extract bias vector
            # use FP equations
            y = W @ x + b               # mat x vec + vec
            x = sigmoid(y)              # apply act func, now next layer    
            acts = np.append(acts,x)    # add to list og activations
        # finished with all layers
        return acts                     # return all activations

    def predict(self,X):
        """ Make predcitions based on feature matrix X """
        Z = np.array([])        # array to hold precited vals
        for x in X:             # each row:
            # run forward pass on sample
            acts = self.forward_pass(x.transpose())
            y = acts[-1]        # last layer activations
            Z = np.append(Z,np.argmax(y))  # add prediction to array
        return Z                # return arr of predictions
            
    def train_model (self,X,y):
        """ train model given feature matrix X & targets y """
        N_samples = X.shape[0]
        Y = self.target_matrix(y)
        # iterate by row:
        for sample in range(N_samples):
            # Forward pass & compute loss func
            pass_acts = self.forward_pass(X[sample].transpose())
            x = acts[-1]                    # last layer activations 
            loss = RSS_Loss(x,Y[sample])    # compute loss
            self.losses = np.append(self.losses,loss)
            # back propagate
