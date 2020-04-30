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

    CLF_MODEL = MLP_utils.MLP_Classifier('Multilayer Perceptron',layer_sizes=(6,),
                                      momentum=1,n_features=4,n_classes=3)

    print(CLF_MODEL.weight_dims)
    print(CLF_MODEL.bias_dims)

    CLF_MODEL = CLF_MODEL.train_model(X,y)  # train the model

    CLF_MODEL.print_weights()

    y_pred = CLF_MODEL.predict(X)
    MLP_utils.confusion_matrix(CLF_MODEL,y,y_pred,show=True)


    #plt.plot(CLF_MODEL.losses)
    #plt.show()