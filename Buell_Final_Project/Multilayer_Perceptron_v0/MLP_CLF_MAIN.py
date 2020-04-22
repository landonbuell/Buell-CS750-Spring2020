"""
Landon Buell
Marek Petrik
CS 750.01
18 April 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_iris

import MLP_CLF_utilities as MLP_utils

            #### MAIN Executable ####

if __name__ == '__main__':
    
    IRIS = load_iris()   
    X,y = IRIS['data'],IRIS['target']
    print("X Shape:",X.shape)
    print("Classes:",np.unique(y).shape)

    CLF_MODEL = MLP_utils.MLP_Classifier('JARVIS',layer_sizes=(10,),
                                      n_features=4,n_classes=3)

    CLF_MODEL = CLF_MODEL.train_model(X,y)  # train the model

    y_pred = CLF_MODEL.predict(X)
    MLP_utils.confusion_matrix(CLF_MODEL,y,y_pred,show=True)
