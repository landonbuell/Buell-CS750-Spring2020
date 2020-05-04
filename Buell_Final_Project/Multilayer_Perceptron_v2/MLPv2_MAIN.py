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

import MLPv2_Utilities as MLPv2_utils

""" This procedure is loosely adapted from James Loy, 'Neural Network Projects with Python'
        Please see written report for citations and more details """

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    
    # TRAINING DATA SET
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])

    momentums = [-0.1,+0.1,+0.5,0.75,+0.9,+1.0,+5.0,+10.0]  # momentum values to use
    outputs = np.array([])


    plt.figure(figsize=(20,8))
    plt.title("Multilayer Perceptron Regressor Losses",size=40,weight='bold')
    plt.xlabel("Iteration",size=20,weight='bold')
    plt.ylabel("Loss Function value",size=20,weight='bold')

    for I in range (len(momentums)):
        MODEL = MLPv2_utils.Multilayer_Perceptron_Regressor('JARVIS',n_features=3,n_classes=1,
                                                            layer_size=10,momentum=momentums[I],max_iters=1000)
        MODEL = MODEL.Train_model(X,y)
        outputs = np.append(outputs,MODEL.output)
        print("Final Loss:",MODEL.losses[-1])
        plt.plot(np.arange(0,1000),MODEL.losses,
                 label='Momentum = '+str(momentums[I]))

    plt.tight_layout()   
    plt.legend(loc='center right')
    plt.grid()
    plt.show()

    print(outputs.reshape(8,-1))