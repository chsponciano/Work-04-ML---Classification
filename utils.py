import scipy.io as scipy


def read(id_data_set : int = 0) -> list:
    return scipy.loadmat(f'data/grupoDados{id_data_set}.mat')