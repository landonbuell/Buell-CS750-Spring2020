"""
Landon Buell
Marek Petrik
CS 750.01
30 April 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import MLPv1_utilities as MLPv1_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    
    # Load in IRIS Data set
    IRIS = load_iris()   
    X,y = IRIS['data'],IRIS['target']
    print("X Shape:",X.shape)
    print("Classes:",np.unique(y).shape)

    # Split Train / Test Data sets
    X_train,X_test,y_train,y_test = \
        train_test_split(X,y,test_size=0.2)

    # Create and train Classifier Instance
    CLF_MODEL = MLPv1_utils.Multilayer_Perceptron_Classifier('JARVIS',
                    n_features=4,n_classes=3,layer_sizes=(10,12),
                    momentum=0.1,batch_size=100,max_iters=10)
    CLF_MODEL = CLF_MODEL.train_model(X_train,y_train)

    plt.plot(CLF_MODEL.losses)
    plt.show()

    for L in range (0,3):
        print(CLF_MODEL.weights[L])
        print(CLF_MODEL.biases[L])


    # Run predicitons
    predictions = CLF_MODEL.prediction(X_test)

    CLF_MODEL = MLPv1_utils.confusion_matrix(CLF_MODEL,y_test,predictions,show=False)
    print(CLF_MODEL.confusion_matrix)
