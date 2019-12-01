from knn_utils import *


# Students: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva

def execute():
    print('\nQuestão 3')
    _data = read('grupoDados3')
    _grupoTrain, _trainRots, _grupoTest, _testRots = separate_dataset(_data)

    # Q3.1: Aplique o kNN ao problema usando k = 1. Qual é a acurácia na classificação?
    # A acurácia é de 58%
    print(f'Q3.1. Acurácia do K = 1: {accuracy(predict(_grupoTrain, _trainRots, _grupoTest, 1), _testRots)}')

    # Q3.2: A acurácia pode ser igual a 92% com o kNN. Descubra por que o resultado atual é muito menor.
    # Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 92% e explique o que você fez e
    # por quê?
    # Foi necessário ir incrementando K para encontrar a acuracia de 92%, onde K deve ser igual a  7
    print(f'Q3.2. k = {get_quantity_groups(_grupoTrain, _trainRots, _grupoTest, _testRots, 0.92)}')