

# Code Cleanup for Data 558
This module provides functions to support a K-nearest neighbors classification of a given dataset.  

A K-nearest neighbors classifier was discussed early in class, but we never bothered to try to implement a solution other than by importing from scikit-learn. So, for this assignment, I implemented a simple classifier that allows K to be variable and which returns predictions and accuracy.

This code presumes that that the sample dataset (X) contains columns of numeric features of similar unit types that are appropriate for KNN measurement. If you were to put in a feature with a different measure than the other features, such as kilometers vs. inches, or classes or string labels, etc...I wouldn't expect too much accuracy or success. But it should work for datasets comprised of with comparable numeric features.

This implementation was influenced by scikit-learns own implementation (https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/neighbors) and is comparable to it in performance. Included are routines to compare this KNN code to scikit-learn's KNeighborsClassifier, and typically they perform similarly. Also, useful was going through Jason Brownlee's tutorials on KNN available on https://machinelearningmastery.com/start-here/. However, this code adds some new items such as using scikit-learns Iris dataset, numpy array basing, and plotting to show Euclidean distance.

## Source code
* **knn_our.py:** The heart of the implementation, containing functions for computing Euclidean distance, sorting nearest neighbors, collecting the vote from each neighbor about which other element is its nearest neighbor, computing accuracy vs. a test set, and making predictions. 

Also, a plot of the Euclidean distance is provided for one (or more than one) of the comparisons, showing the K items circled which will be included in the voting. If you want to see every row plotted, you can find the plt.show() line in getNeighbors and uncomment it, but that does produce one plot per row which is a lot. Default is one plot per run, with the last row being what's shown. 
* **knn_skl.py:** This code simply returns a scikit-learn KNeighborsClassifier for comparison to our implementation.
* **knn_iris.py:** Run this code to measure accuracy against the famous Iris dataset, as provided by scikit-learn.
* **knn_random.py:** Run this code to measure accuracy agaist a randomly generated dataset, as provided by scikit-learn's make_classfication.
* **knn_data.py** This module contains functions to load and standardize the datasets.

## Use
Make sure all files are downloaded and available in your Python path/environment for importing, as they cross-reference each other.

Run knn_random in Python 3 to launch the method on a simple simulated dataset, visualize the training process, and print the performance. You can change K to see different neighbor counts.
  
Run knn_iris in Python 3 to launch the method on a real-world dataset of your choice, visualize the training process, and print the performance. You can change K to see different neighbor counts.

Either of the above includes a comparison to the accuracy of sci-kit learn's KNeighborsClassifier. Re-running knn_random should show that usually they are similar.

In knn_our you can find and comment/uncomment the two plt.show() calls to choose to show one plot for a run or one plot for each comparison. Default as checked in is one plot total not one per row.



