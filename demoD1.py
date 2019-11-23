import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats

def read() -> list:
    return scipy.loadmat(f'data/grupoDados1.mat')

def knn(grupoTrain : np, trainRots : np, grupoTest : np, k : int) -> np:
    _labels = np.zeros([grupoTest.shape[0], trainRots.shape[1]])

    for i in range(grupoTest.shape[0]):
        _copy_test  = np.tile(grupoTest[i,:], (grupoTrain.shape[0], 1))
        _dists      = dist(_copy_test, grupoTrain)
        _neighbors  = np.argsort(_dists)[:k]
        _swat       = trainRots[_neighbors]       
        _labels[i]  = _swat.sum(axis=0)

    return _labels

def accuracy(predicted_label : np, testRots : np) -> float:
    return float(sum(predicted_label == testRots) / len(testRots))

def dist(x : np, y : np) -> float:
    return np.sqrt(np.power(x - y, 2).sum(axis=1))

def normalization(data : list, minmax : list) -> list:
    return [(line[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) for i in range(len(data[0])) for line in data]
        

def plot(data : np, labels : np, d1 = 1, d2 = 2):
    fig = plt.figure()
    plt.suptitle(f'KNN')
    plt.plot(data[labels == 1][d1], data[labels == 1][d2], color='red', marker='^')
    plt.plot(data[labels == 2][d1], data[labels == 2][d2], color='blue', marker='+')
    plt.plot(data[labels == 3][d1], data[labels == 3][d2], color='green', marker='o')
    
if __name__ == "__main__":
    _data_set    = read()

    _grupoTrain  = np.asarray(_data_set['grupoTrain'])
    _trainRots   = np.asarray(_data_set['trainRots'])
    _grupoTest   = np.asarray(_data_set['grupoTest'])
    _testRots    = np.asarray(_data_set['testRots'])

    _predicted_label = knn(_grupoTrain, _trainRots, _grupoTest, 1)
    _accuracy        = accuracy(_predicted_label, _testRots)
    print(_accuracy)

    assert (_accuracy ==  0.96), "The accuracy of K = 1 must be 96%."

    _predicted_label = knn(_grupoTrain, _trainRots, _grupoTest, 10)
    _accuracy        = accuracy(_predicted_label, _testRots)
    
    print(_accuracy)

    assert (_accuracy ==  0.94), "The accuracy of K = 10 must be 94%."
