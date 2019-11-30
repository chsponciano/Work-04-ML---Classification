from knn_utils import *
import demoD1
import demoD2
import demoD3


# Students: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva

if __name__ == "__main__":
    print('Inicio da calibragem')
    _data = read('grupoDados1')
    _grupoTrain, _trainRots, _grupoTest, _testRots = separate_dataset(_data)

    _predicted_label = predict(_grupoTrain, _trainRots, _grupoTest, 1)
    _accuracy        = accuracy(_predicted_label, _testRots)
    assert (_accuracy ==  0.96), 'The accuracy of K = 1 must be 96%.'

    _predicted_label = predict(_grupoTrain, _trainRots, _grupoTest, 10)
    _accuracy        = accuracy(_predicted_label, _testRots)
    assert (_accuracy ==  0.94), 'The accuracy of K = 10 must be 94%.'
    print('Fim da calibragem')    

    demoD1.execute()
    demoD2.execute()
    demoD3.execute()