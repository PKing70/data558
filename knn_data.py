"""
File: knn_data.py
Created on Thu May 31 23:19:55 2018
@author: pking70@uw.edu

"""
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def loadIrisData(test_size):
    """
    Loads iris data from scikit-learn's load_iris.
    Parameters:      
        test_size    The ratio of test data to train data upon which you want
                     to split. 0.25 = one quarter of the dataset becomes test 
                     set.
    Returns:       
        X_train      The training samples, N by F, with N=rows of samples
                     and F=columns of features.
        y_train      A training set of labels corresponding to X_train with 
                     len(y_train)=N.
        X_test       The test samples, M by F, with M=rows of samples and 
                     F=columns of features.
        y_test       A test set of labels corresponding to X_test with 
                     len(y_test)=M.
    """
    from sklearn.datasets import load_iris
    
    # Load sklearn's iris set
    iris_dataset = load_iris()

    # Split it up
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0, 
        test_size=0.25)
    
    # Standardize it
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    print('Iris data')
    
    return X_train, X_test, y_train, y_test

def loadRandomData(test_size):
    """
    Loads random data from scikit-learn's make_classfiication.
    Parameters:      
        test_size    The ratio of test data to train data upon which you want
                     to split. 0.25 = one quarter of the dataset becomes test 
                     set.
    Returns:       
        X_train      The training samples, N by F, with N=rows of samples
                     and F=columns of features.
        y_train      A training set of labels corresponding to X_train with 
                     len(y_train)=N.
        X_test       The test samples, M by F, with M=rows of samples and 
                     F=columns of features.
        y_test       A test set of labels corresponding to X_test with 
                     len(y_test)=M.
    """
    from sklearn.datasets import make_classification
    
    # Generate a random set of data
    X, y = make_classification(n_samples=150, n_features=4, n_informative=4, 
                               n_redundant=0, n_classes=3)
    
    # Split it up
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, 
                                                    test_size=0.25)
    # Standardize it
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    print('Random data')
    
    return X_train, X_test, y_train, y_test

