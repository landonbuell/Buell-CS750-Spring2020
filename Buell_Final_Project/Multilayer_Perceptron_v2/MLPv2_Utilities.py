"""
Landon Buell
Marek Petrik
CS 750.01
18 April 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

            #### ACTIVATION FUNCTIONS ####

def sigmoid (z):
    """ Logisitc Sigmoid Activation Function """
    return 1 / (1.0 + np.exp(-z))

def sigmoid_deriv (z):
    """ Derivative of Logisitc Sigmoid Activation Function """
    return z * (1.0 - z)

            #### OBJECT DEFINITIIONS ####

class Multilayer_Perceptron_Regressor():
    """
    Create 1-Hidden Layer Multilayer Perceptron Regressor Object
    --------------------------------
    name (str) : Name at attach to MLP for user use
    n_features (int) : Number of features for network
    n_classes (int) : Number of output target classes - 1 for Regressors
    layer_size (int) : Number of neuron in single hidden layer
    momentum (float) : Momentum value to scale gradients by
    max_iters (int) : Indicate maximum number of training iters
    --------------------------------
    Return initialied instance of Multilayer Perceptron Regressor 
    """

    def __init__(self,name,n_features,n_classes=1,layer_size=10, 
                 momentum=0.9,max_iters=1000):
        """ Initialize Object Instance """
        self.name = name                    # name of network       
        self.n_features = n_features        # number of input features
        self.n_classes = n_classes          # number of output neurons (1 for regression)
        self.layer_size = layer_size        # sizes of hidden layers
        self.momentum = momentum            # scalar for SGD
        self.max_iters = max_iters          # iterations for training
        self.losses = np.array([])          # hold loss functions      
        self = self.generate_weights()      # make weight matrices

    def generate_weights (self):
        """ Generate 3 weight matrices """
        self.W0 = np.random.rand(self.n_features,self.layer_size)
        self.W1 = np.random.rand(self.layer_size,self.n_classes)
        return self         

    def RSS_loss (self):
        """ Compute loss for prediction """
        loss = np.sum((self.output - self.y )**2)
        self.losses = np.append(self.losses,loss)
        return self

    def forward_pass (self):
        """ Pass Design Matrix through network """
        self.layer1 = sigmoid(np.dot(self.X,self.W0))
        self.output = sigmoid(np.dot(self.layer1,self.W1))
        return self

    def back_propagate(self):
        """ Compute gradient w.r.t to Weights matrices """
        # compute gradients
        dx2 = 2*(self.y-self.output) * sigmoid_deriv(self.output)
        dW1 = np.dot(self.layer1.T , dx2)
        dx1 = np.dot(dx2 , self.W1.T) * sigmoid_deriv(self.layer1)
        dW0 = np.dot(self.X.T , dx1 )

        # Update matrices w/ gradients
        self.W0 += (dW0 * self.momentum)
        self.W1 += (dW1 * self.momentum)

        return self

    def Train_model(self,X,y):
        """ Train model with desgin matrix X, target vector y """
        self.X = X      # set design matrix
        self.y = y      # set feature matrix
        # Execute forward pass
        for I in range (self.max_iters):
            self.forward_pass()     # compute forward pass
            self.RSS_loss()         # updat loss array
            self.back_propagate()   # compute grads
        return self


