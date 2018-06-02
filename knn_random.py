"""
File: knn_random.py
Created on Thu May 31 23:19:55 2018
@author: pking70@uw.edu

This module compares the results of knn_our to sklearn's KNeighbors classifier.
This loads randomly generated data.

Run it repeatedly with random data to see that sometimes our code is better and
sometimes it is worse regarding accuracy.

Variables:
    test_size    The split ratio of test data to train data.
    X_train      The training samples from Iris or random data.
    y_train      A training set of labels from Iris or random data.
    X_test       The test samples from Iris or random data.
    y_test       A test set of labels from Iris or random data.
    k            The number of neighbors to consider.
    accuracy     The accuracy of our KNN computation
    sklknn       The classifier returned from scikit-learns KNN classification,
                 from which we can get accuracy via its score method.
"""
from knn_data import loadRandomData
from knn_our import ourKnn
from knn_skl import sklKnn

test_size = 0.25    # split ratio: 0.25=one quarter of dataset is set for test

# Load data 
X_train, X_test, y_train, y_test = loadRandomData(test_size)

# How many neighbors?
k=3

# Get prediction accuracy from both ways
accuracy = ourKnn(X_train, y_train, X_test, y_test, k)
sklknn = sklKnn(X_train, y_train, k)

# Compare results
print('Our accuracy: ' + repr(round(accuracy, 2)) + ' %')
print('Scikit-learn accuracy: {:.2f}'.format(100*sklknn.score(X_test, 
      y_test)), '%')