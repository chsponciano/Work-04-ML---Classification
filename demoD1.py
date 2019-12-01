from knn_utils import *


# Students: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva

def execute():
    print('\nQuestão 1')
    _data = read('grupoDados1')
    _grupoTrain, _trainRots, _grupoTest, _testRots = separate_dataset(_data)

    _predicted_label_k1 = predict(_grupoTrain, _trainRots, _grupoTest, 1)
    _accuracy        = accuracy(_predicted_label_k1, _testRots)
    assert (_accuracy ==  0.96), 'The accuracy of K = 1 must be 96%.'

    _predicted_label_k10 = predict(_grupoTrain, _trainRots, _grupoTest, 10)
    _accuracy        = accuracy(_predicted_label_k10, _testRots)
    assert (_accuracy ==  0.94), 'The accuracy of K = 10 must be 94%.'
    print('Fim da calibragem')  

    # Q1.1. Qual é a acurácia máxima que você consegue da classificação?
    # O máximo atingido é de 0,98 de acurácia sendo K = 3
    _best_accuracy, _best_k, _best_prediced = calculate_better_accuracy(_grupoTrain, _trainRots, _grupoTest, _testRots)
    print(f'Q1.1. Acuracia máxima atingida: {_best_accuracy} - K = {_best_k}')

    # Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta
    # classificação?
    # Sim, é necessário ter todas as caracteristicas para obter a acurácia máxima
    for i in range(1, _grupoTrain.shape[1] + 1):
        _best_accuracy, _best_k, _ = calculate_better_accuracy(_grupoTrain[:, :i], _trainRots, _grupoTest[:, :i], _testRots)
        print(f'Q1.2. Acuracia máxima atingida com {i} caracteristicas: {_best_accuracy} - K = {_best_k}')

    plot(_data['grupoTest'], [_testRots, _predicted_label_k1, _predicted_label_k10, _best_prediced], ['Original', 'K = 1', 'K = 10', f'Acurácia máxima - K = {_best_k}'], 'Questão 1')