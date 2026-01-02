# Name : Masoud Rafiee
# Date : Feb, 1st 2025
# Description :  Intelligent Systems and Neural Networks - CS446
# Professor : Dr. Rachid Hedjam
# Assignment 1 : Solve a problem of face recognition using Multi-layer Perceptron (MLP)
#######################################################################################
# Load Libraries :
from hmac import digest_size

#various dataset from this
import sklearn.datasets as ds
#for interactive visualization (works like MATLAB)
from matplotlib import pyplot as plt
#For Scientific computing (Multi-Dimensional Arrays, matrices, high level math functions)
import numpy as np
import random
import os
#for implementing multi layer perceptron algorithms for classification
from sklearn.neural_network import MLPClassifier as MLP
#for computing the accuracy of classification score
from sklearn.metrics import accuracy_score
#for spliting arrays/matrices of data into random train and test subsets
from sklearn.model_selection import train_test_split
# for returning the indices of the max values along an axis
from numpy.ma.core import argmax
#to ignore any warning messages: so they don't affect the functionality of the program
import warnings
warnings.filterwarnings('ignore')
########################################
# Load Face Dataset

#load the Olivetti faces dataset from sklearn.datasets
faces = ds.fetch_olivetti_faces()
#assign the data (features) of the dataset to the X
X=faces.data
#Assign the target (labels) of the dataset to y
y=faces.target
#showing the number of features in X and number of samples in y
#400 target labels , and 400 samples each with 4096 features (pixels)
print(X.shape, y.shape)
#################################
# Display one random face:

#generate a random index from the array of features
random_index = random.randint(0,len(X)-1)
#representing it as 64x64 pixel grid
plt.imshow(X[random_index].reshape(64, 64), cmap='gray')
plt.show()
#################################
# Split Data into Training and Test Sets

#spliting dataset into train/test: 0.3 means 30% of data will be used for testing, 70% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .3, random_state=42)
#280 samples in the training set, 120 samples in test set, each with 4096 features
print (X_train.shape, X_test.shape)
##################################
# Define Parameter Search Space / setting

mlp_parameters = {
    # three hidden layers with 100, 200, 100 neurons (respectively)
    'hidden_layer_sizes': [(100,200,100), (100,), (50,100,50,10)],
    # defining the activation functions in hidden layers
    'activation': ['relu', 'tanh', 'logistic'],
    # define optimization algorithm used to find the weights that minimizes the loss function
    #stochastic gradient descent (sgd) and 'adam'
    'solver':['sgd','adam'],
    # defining initial learning rates in the optimization agorithms
    'learning_rate_init':[1E-2,1E-3],
}
####################################
# Perform Grid Search : Get the best parameter setting and the best accuracy

# for each combination of parameters -> create an MLP classifier -> train it on training set
# -> predict the labels of the test set. computes accuracy
# -> store the accracy with the correspondece parameter setting
# in a dictionary (result)
results = {}
for hls in mlp_parameters['hidden_layer_sizes']:
    for act in mlp_parameters ['activation']:
        for solv in mlp_parameters['solver']:
            for lri in mlp_parameters['learning_rate_init']:
                clf = MLP(hidden_layer_sizes=hls, activation=act, solver=solv, learning_rate_init=lri, max_iter=1000)
                clf.fit(X_train, y_train)
                y_pred=clf.predict(X_test)
                acc=accuracy_score (y_test, y_pred)
                param_setting=(hls,act,solv,lri)
                results[param_setting]=acc
#######################################
# Find the best parameter setting:

best_param_setting = max(results, key=results.get)
best_accuracy = results[best_param_setting]
print("Best Accuracy: ", best_accuracy)
print("Best Parameter: ", best_param_setting)
########################################
# Create and Train MLP with Best Parameters

#creating an MLP classifier with the best parameter setting
best_clf = MLP(hidden_layer_sizes=best_param_setting[0], activation=best_param_setting[1], solver=best_param_setting[2], learning_rate_init=best_param_setting[3], max_iter=1000)
#train the classifier on the training set
best_clf.fit(X_train, y_train)
##########################################
# Test the Model

#predict the labels of the test set
y_pred = best_clf.predict(X_test)
#compute the accuracy of the predicition
acc=accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
###########################################
# Find Misclassified Samples

#finding indices of misclassified samples
missclassified_indices = np.where(y_test!=y_pred)[0]
print("Indexes of misclassified samples: ", missclassified_indices)
num_misclassified = len(missclassified_indices)
print("Number of Misclassified samples: ", num_misclassified)
###########################################
# Display Misclassified Face Images

#creating a figure with subplots displaying the true and misclassified face images
fig, axes = plt.subplots(nrows=num_misclassified, ncols=2, figsize=(10, 5*num_misclassified))
#running thro misclassified indices and dispalys them along with trueimages of them
for i, idx in enumerate (missclassified_indices):
    true_face = X_test[idx].reshape(64,64)
    pred_face = X_test[idx].reshape(64,64)
    axes[i,0].imshow(true_face,cmap='gray')
    axes[i,0].set_title(f'True Label:{y_test[idx]}')
    axes[i,1].imshow(pred_face, cmap='gray')
    axes[i,1].set_title(f'Predicted Label: {y_pred[idx]}')
#adjusting space between subplots
plt.tight_layout()
plt.show()

####
# DONE !
# Thank you!
