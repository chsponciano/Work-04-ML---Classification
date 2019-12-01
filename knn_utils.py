import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats
from scipy.stats import mode

# Students: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva


def read(filename : str) -> tuple:
    return scipy.loadmat(f'data/{filename}.mat')

def separate_dataset(data : np) -> tuple:
    return np.asarray(data['grupoTrain']), np.asarray(data['trainRots']), np.asarray(data['grupoTest']), np.asarray(data['testRots'])

def predict(grupoTrain : np, trainRots : np, grupoTest : np, k : int) -> np:
    _labels = np.zeros([grupoTest.shape[0], trainRots.shape[1]])

    for i in range(grupoTest.shape[0]):
        _copy_test  = np.tile(grupoTest[i,:], (grupoTrain.shape[0], 1))
        _dists      = distance(_copy_test, grupoTrain)
        _neighbors  = np.argsort(_dists)[:k]
        _swat       = trainRots[_neighbors]   
        _labels[i]  = mode(_swat)[0][0]

    return _labels

def accuracy(predicted_label : np, testRots : np) -> tuple:
    return round(float(sum(predicted_label == testRots) / len(testRots)), 2)

def calculate_better_accuracy(grupoTrain : np, trainRots : np, grupoTest : np, testRots : np, k : int = 1, attempts : int = 20) -> tuple:
    _temp_attempts = attempts
    _best_accuracy = _best_knn = 0

    while _temp_attempts != 0:
        _predicted_label = predict(grupoTrain, trainRots, grupoTest, k)
        _accuracy        = accuracy(_predicted_label, testRots)

        if _accuracy > _best_accuracy:
            _best_accuracy  = _accuracy
            _best_knn       = k
            _temp_attempts  = attempts
        else:
            _temp_attempts -= 1
        
        k += 1

    return _best_accuracy, _best_knn

def get_quantity_groups(grupoTrain : np, trainRots : np, grupoTest : np, testRots : np, accuracy_search : float, k : int = 1, attempts : int = 20):
    while k != attempts:
        _predicted_label = predict(grupoTrain, trainRots, grupoTest, k)
        _accuracy        = accuracy(_predicted_label, testRots)

        if (_accuracy == accuracy_search):
            return k
        else:
            k += 1
    
    return 0

def distance(p : np, q : np) -> float:
    return np.sqrt(np.power(p - q, 2).sum(axis=1))

def normalization(data : list) -> list:
    _buffer_normalization = np.ndarray(shape=data.shape)
    
    for idx in range(data.shape[1]):
        _element = data[:, idx]
        _buffer_normalization[:, idx] = (_element - np.min(_element)) / (np.max(_element) - np.min(_element))

    return _buffer_normalization

def get_label_data(data : list, labels : list, current_label : float, index : int) -> list:
    _buffer = list()

    for idx in range(len(data)):
        if  labels[idx] == current_label:
            _buffer.append(data[idx][index])
            
    return _buffer
        
def plot(data : list, labels : list, d1 : int = 0, d2 : int = 1):
    fig, ax = plt.subplots()
    plt.suptitle(f'KNN')
    ax.scatter(get_label_data(data, labels, 1, d1), get_label_data(data, labels, 1, d2), c='red' , marker='^')
    ax.scatter(get_label_data(data, labels, 2, d1), get_label_data(data, labels, 2, d2), c='blue' , marker='+')
    ax.scatter(get_label_data(data, labels, 3, d1), get_label_data(data, labels, 3, d2), c='green', marker='.')
    plt.show()
    