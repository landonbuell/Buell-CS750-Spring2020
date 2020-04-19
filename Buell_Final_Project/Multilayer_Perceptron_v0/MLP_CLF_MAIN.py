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

import MLP_CLF_utilities as MLP_utils

            #### MAIN Executable ####

if __name__ == '__main__':
    
    CLF_MODEL = MLP_utils.MLP_Classifier('JARVIS',layer_sizes=(4,6,8,10),
                                      n_features=2,n_classes=1)

    X = np.array([[0,0],[0,1],[1,0],[1,1]])

    predictions = CLF_MODEL.predict(X)
