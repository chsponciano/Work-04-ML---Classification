from knn_utils import *


# Students: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva

def execute():
    print('\nQuestão 2')
    _data = read('grupoDados2')
    _grupoTrain, _trainRots, _grupoTest, _testRots = separate_dataset(_data)

    # Q2.1: Aplique seu kNN a este problema. Qual é a sua acurácia de classificação?
    # A acurácia máxima é de 78,33%
    _best_accuracy, _best_k, _best_prediced = calculate_better_accuracy(_grupoTrain, _trainRots, _grupoTest, _testRots)
    print(f'Q2.1. Acuracia máxima atingida: {_best_accuracy:.2f} - K = {_best_k}')

    # Q2.2: A acurácia pode ser igual a 98% com o kNN. Descubra por que o resultado atual é muito menor.
    # Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 98% e explique o que você fez e 
    # por quê?
    # O motivo da acurácia estar muito baixo, é por causa da grande divergencia entre os valores.
    # Para encontrarmos o valor de 98% foi necessário efetuar a normalização dos grupos de teste e treino 
    _k, _prediced_normalization = get_quantity_groups(normalization(_grupoTrain), _trainRots, normalization(_grupoTest), _testRots, 0.98, attempts=_grupoTrain.shape[0])
    print(f'Q2.2. Dataset normalizado - Acurácia = 98% - K = {_k}')

    plot(_data['grupoTest'], [_testRots, _best_prediced, _prediced_normalization], ['Original', f'Sem Normalização, K = {_best_k}', f'Com Normalização, K = {_k}'], 'Questão 2')