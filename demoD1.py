import utils
from math import sqrt


def calculate_knn(grupoTrain : list, trainRots : list, grupoTest : list, k : int):
    # %Para cada exemplo de teste
    # % Calcule a distância entre o exemplo de teste e os dados de treinamento
    _dist = dist(grupoTest)


    # % Ordene as distâncias. A ordem iX de cada elemento ordenado é
    # importante:
    # % [distOrdenada ind] = sort(...);
    # % O rótulo previsto corresponde ao rótulo do exemplo mais próximo (iX(1))
    pass

def dist(grupoTrain : list, grupoTest : list) -> float:
    return sqrt(sum([(test - train) ** 2 for test, train in zip(grupoTest, grupoTrain)]))

def normalization():
    pass

def plot():
    pass


if __name__ == "__main__":
    _data_set   = utils.read(1)
    grupoTrain  = _data_set['grupoTrain']
    trainRots   = _data_set['trainRots']
    grupoTest   = _data_set['grupoTest'] 
    testRots    = _data_set['testRots']


