from knn_utils import *


# Students: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva

def execute():
    print('\nQuestão 1')
    _data = read('grupoDados1')
    _grupoTrain, _trainRots, _grupoTest, _testRots = separate_dataset(_data)

    # Q1.1. Qual é a acurácia máxima que você consegue da classificação?
    # O máximo atingido é de 0,98 de acurácia sendo K = 3
    _best_accuracy, _best_k = calculate_better_accuracy(_grupoTrain, _trainRots, _grupoTest, _testRots)
    print(f'Q1.1. Acuracia máxima atingida: {_best_accuracy} - K = {_best_k}')
    
    # Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta
    # classificação?
    # Sim, é necessário ter todas as caracteristicas para obter a acurácia máxima
    for i in range(1, _grupoTrain.shape[1] + 1):
        _best_accuracy, _best_k = calculate_better_accuracy(_grupoTrain[:, :i], _trainRots, _grupoTest[:, :i], _testRots)
        print(f'Q1.2. Acuracia máxima atingida com {i} caracteristicas: {_best_accuracy} - K = {_best_k}')
    
    print(_trainRots[predict(_grupoTrain, _trainRots, _grupoTest, 3)])
    _labels = [_testRots, _trainRots]
    plot(_grupoTest, _labels)
