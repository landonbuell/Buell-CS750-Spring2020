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

class Multilayer_Perceptron_Classifier():
    """
    Create 2-Hidden Layer Multilayer Perceptron Classifier Object
    --------------------------------
    name (str) : Name at attach to MLP for user use
    n_features (int) : Number of features for network
    n_targets (int) : Number of target classes
    batch_size (int) : Number of samples in a single mini-batch
    max_iters (int) : Indicate maximum number of training iters
    --------------------------------
    Return initialied instance
    """

    def __init__(self,name,n_features,n_classes,
                 layer_sizes = (10,10), momentum=0.9,
                 batch_size=100,max_iters=100):
        """ Initialize Object Instance """
        self.name = name                    # name of network       
        self.depth = len(layer_sizes)+2     # all layers
        self.n_features = n_features        # number of input features
        self.n_classes = n_classes          # number of output classes
        self.layer_sizes = layer_sizes      # sizes of hidden layers
        self.momentum = momentum            # scalar for SGD
        self.batch_size = batch_size        # sizes of mini-batches for training
        self.max_iters = max_iters          # maximum iterations over data
        self.losses = np.array([])          # hold loss functions
        # Generate weight matrices & bias vectors
        self.weights = self.generate_weights()  # make weight matrices
        self.biases = self.generate_biases()    # make bias vectors

    def generate_weights (self):
        """ Generate 3 weight matrices """
        W0 = np.random.rand(self.layer_sizes[0],self.n_features)
        W1 = np.random.rand(self.layer_sizes[1],self.layer_sizes[0])
        W2 = np.random.rand(self.n_classes,self.layer_sizes[1])
        return [W0,W1,W2]               # return weight matrices

    def generate_biases (self):
        """ Generate 3 bias vectors """
        b0 = np.random.rand(self.layer_sizes[0],1)
        b1 = np.random.rand(self.layer_sizes[1],1)
        b2 = np.random.rand(self.n_classes,1)
        return [b0,b1,b2]               # return bias vectors

    def RSS_loss (self,x,y):
        """ Compute loss for prediction """
        return np.sum((x-y)**2)         # RSS loss 


    def target_matrix (self,y):
        """ Construct target matrix Y from vector y """
        y = y.ravel()               # flatten y
        n_samples = y.shape[0]      # samples in y
        Y = np.zeros((n_samples,self.n_classes),dtype=float) # init matrix
        for I in range (n_samples):     # in each sample
            target = int(y[I])          # target for sample
            Y[I][target] += 1.0         # make into 1
        return Y                    # return target matrix

    def forward_pass (self,x):
        """ Forward pass data sample x through network 
            Modified from Goodfellow, 'Deep Learning', algorithm 6.3 """
        x = x.reshape(-1,1)     # reshape to column vector
        FIN_ACTS = [x]           # layer activations
        PRE_ACTS = []           # pre-activation function
        for l in range (0,self.depth-1):   #  0-3 layers (W & b)
            # Get Layer weight & baises
            W,b = self.weights[l],self.biases[l]
            # Compute next-layer
            a = W @ x + b           # pre-activation
            x = sigmoid(a)          # apply sigmoid
            # add values to list
            PRE_ACTS.append(a)
            FIN_ACTS.append(x)
        self.fin_acts = FIN_ACTS    # set attrb
        self.pre_acts = PRE_ACTS    # set attrb
        return x.ravel()            # return final layer - FLATTENED

    def prediction (self,X):
        """ Makes prediction based on matrix X (n_samples x n_features) """
        n_samples = X.shape[0]          # number of predictions to make
        predictions = np.array([])      # array to hold predicted values
        for I in range (n_samples):     # each row
            y = self.forward_pass(X[I].reshape(-1,1))   # pass arr through network
            predicted_class = np.argmax(y)              # pred is idx of max val
            predictions = np.append(predictions,predicted_class)
        return predictions

    def train_model (self,X,y):
        """ Train Model baed on matrix X and targt y """
        n_samples = X.shape[0]      # number of samples to train with
        Y = self.target_matrix(y)   # make target matrix   
        
        for I in range (0,self.max_iters):  # iterate through data
            print('iteration:',I)

            # Each Sample in Data Set"
            for sample,target in zip(X,Y):
                # Forward Pass Data
                x = self.forward_pass(sample.reshape(-1,1))

                # Compute Weight & Bias Gradients
                weight_grads,bias_grads = \
                    self.back_prop(target)
           
                # update weights & biases based on mini-batch grads
                for l in range(self.depth-1):           # each W & b arr
                    self.weights[l] += weight_grads[l]     # update W
                    self.biases[l] += bias_grads[l]        # update b
                # update loss list
                self.losses = np.append(self.losses,self.RSS_loss(x,target))

        return self                 # return fitted self inst.


    def mini_batch (self,X_batch,Y_batch):
        """ Pass mini-batch (UNUSED) """
        avg_batch_loss = 0          # loss for mini-batch
        batch_weight_grads = [np.zeros(W.shape) for W in self.weights]   
        batch_bias_grads = [np.zeros(b.shape) for b in self.biases]
        for I in range (self.batch_size):       # each sample in mini-batch:
            # Forward pass data 
            x = self.forward_pass(X_batch[I].reshape(-1,1)) # NN output
            y = Y_batch[I].ravel()                          # target output
            # Back Progagate
            sample_weight_grads,sample_bias_grads = \
                self.back_propagate(y)
            for l in range(self.depth-1):
                batch_weight_grads[l] += sample_weight_grads[l]
                batch_bias_grads[l] += sample_bias_grads[l]
            # compute sample loss
            avg_batch_loss += self.RSS_loss(x,y)        # sample loss
        # Iterated through full mini-batch, divide by batch size
        avg_batch_loss /= self.batch_size           # averge loss for this batch
        for l in range(self.depth-1):               # each W & b arr
            batch_weight_grads[l] *= self.momentum/self.batch_size    
            batch_bias_grads[l] *= self.momentum/self.batch_size  
        # return loss, grad weights & grad biases
        return avg_batch_loss,batch_weight_grads,batch_bias_grads

    def back_prop (self,y):
        """ Compute the gradients for each layer per single sample output
            Modified from Goodfellow, 'Deep Learning', algorithm 6.4 """
        x = self.fin_acts[-1].reshape(-1,1)     # network output
        y = y.reshape(-1,1)                     # expected output
        dx = 2*(x - y)                          # grad w.r.t last layer 
        weight_grads = [np.zeros(W.shape) for W in self.weights]   # hold weight grads
        bias_grads = [np.zeros(b.shape) for b in self.biases]       # hold bias grads
        for l in range (self.depth-2,0,-1):                         # iter through layers
            # Comptue grad w.r.t to bias & weights
            db = -dx * sigmoid_deriv(self.pre_acts[l])   
            dW = -db @ self.fin_acts[l].reshape(1,-1)    
            # Add changes to grad arrays
            weight_grads[l] += dW
            bias_grads[l] += db
            # Compute new grad w.r.t layer
            dx = self.weights[l].transpose() @ db
        # Return grad for sample
        return weight_grads,bias_grads  

    def back_propagate (self,y):
        """ Compute the gradients for each layer per single sample output
            Will hard-code this function for proof of concept """
        x = self.fin_acts[-1].reshape(-1,1)     # network output
        y = y.reshape(-1,1)                     # expected output
        dx = 2*(x - y)                          # grad w.r.t last layer 
        pass

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