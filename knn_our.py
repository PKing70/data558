"""
File: knn_our.py
Created on Thu May 31 23:19:55 2018
@author: pking70@uw.edu

This module provides functions to support a K-nearest neighbors classification 
of a given dataset.

Our code presumes that X contains columns of numeric features of similar unit
types that are appropriate for KNN measurement. If you were to put in a feature
with a different measure than the other features, such as kilometers vs. 
inches, or classes or string labels, etc...don't expect too much accuracy or 
success. But it should work for datasets with comparable numeric features.
"""
import math
import operator

def euclideanDist(i1, i2):
    """
    Get the Euclidian distance between two sample instances
    Parameters:      
        i1   The first instance of data.
        i2   The second instance of data.
    
    Returns:       
        The Euclidian distance between the two instances, which is the square
        root of the sum of the squared differences. See
        https://en.wikipedia.org/wiki/Euclidean_distance for more info.
    """
    dist = 0
    
    for g in range(i2.shape[0]):          # For each data element,
        dist += pow((i1[g] - i2[g]), 2)   # sum the square of the differences
        
    return math.sqrt(dist)                # then take the square root.

def getNeighbors(X_train, X_test_i, k):
    """
    Get the K nearest neighbors from a dataset to a given test instance sample
    Parameters:      
        X_train    The training dataset.
        X_test_i   An instance of test data to measure against each row of
                   X_train.
        k          The number of neighbors to consider.
        
    Returns:       
        neighbors  The k nearest neighbors from X_train to X_test_i.
    """
    dists = []
    
    for h in range(len(X_train)):             # For each data element,
        dist = euclideanDist(X_train[h], X_test_i)   # get distance to X_test_i
        dists.append((h, dist))               # Add it to a list
    
    dists.sort(key=operator.itemgetter(1))    # Sort the list by distance
    
    neighbors = []
    
    for n in range(k):                        # For the k nearest neighbors
        neighbors.append(dists[n][0])         # Add them to a list
        
    return neighbors 

def getVotes(neighbors, y_train):
    """
    Get the vote from each neighbor to determine which ones are nearer
    Parameters:      
        neighbors   The list of neighbors produced by getNeighbors.
        y_train     The corresponding labels to any given neighbor.
    
    Returns:       
        The results of the voting, sorted by magnitude of distance
    """
    votes = {}
    
    for x in range(len(neighbors)):    # For each neighbor
        vote = y_train[neighbors[x]]   # its vote is its y label
        
        if vote in votes:
            votes[vote] += 1           # increment its vote count
        else:
            votes[vote] = 1            # add it to the vote list
    
    sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), 
                         reverse=True) # Sort votes largest to smallest
    
    return int(sortedVotes[0][0])      # Return as integer the top vote getter

def getAcc(y_test, predictions):
    """
    Get the accuracy of our predictions
    Parameters:      
        y_test         The labels of the test data with which to compare.
        predictions    Our prediction list produced by getVotes.
    
    Returns:       
        The percentage of accurate predictions
    """
    correct = 0
    
    for j in range(len(y_test)):        # For each item in the test set
        if y_test[j] == predictions[j]: # compared to each item in predictions,
	        correct += 1                 # Count when they match
            
    return (correct/float(len(y_test))) * 100.0 # Return as a percentage

def ourKnn(X_train, y_train, X_test, y_test, k):
    """
    Perform a K-nearest neighbors evaluation between a train and test set
    Parameters:      
        X_train    The training dataset.
        y_train    The labels corresponding to the rows of X_train.
        X_test     The dataset to test against.
        y_test     The labels corresponding to the rows of X_test.
        k          The number of neighbors to consider.
    
    Returns:       
        The accuracy of our predictions.
    """
    predictions=[]
    
    for i in range(len(X_test)):              # For each item in X)_test
        neighbors = getNeighbors(X_train, X_test[i,:], k) # Get its k neighbors 
        result = getVotes(neighbors, y_train) # Vote on which are closest
        predictions.append(result)
    
    accuracy = getAcc(y_test, predictions)    # Measure accuracy
    
    return accuracy

