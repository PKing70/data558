"""
File: knn_skl.py
Created on Thu May 31 23:19:55 2018
@author: pking70@uw.edu

This module provides a function, sklKnn, to perform a K-nearest neighbors 
classification of a given dataset using scikit-learn's KNeighborsClassifier 
(http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.
KNeighborsClassifier.html).
"""
def sklKnn(X, y, k):
    """
    Parameters:      
        X    A training dataset of observations, N by F, with N=rows of samples
             and F=columns of features.
        y    A training set of labels corresponding to X with len(y)=N.
        k    The number of neighbors to consider.
    
    Returns:       
        knn  The scikit-learn KNeighbors classifier fit to the provided data.
    """
    from sklearn.neighbors import KNeighborsClassifier

    # Fit the data using scikit-learn's KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    
    return knn