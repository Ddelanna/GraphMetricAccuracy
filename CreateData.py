import numpy as np
import pandas as pd
from HelperFunctions import _set_random_state

def _process_data(data, labels):
    """ Ensures that the data and labels are of the correct type and have the same indexing to
        make querying easier later on. """
    labels = pd.Series(labels, index=np.arange(data.shape[0]))

    data = pd.DataFrame(data, index=np.arange(data.shape[0]))
    data['query index'] = labels.index
    data.set_index('query index', inplace=True)

    return data, labels


def create_spiral_data(num_points, dimension=2, random_state=None):
    _random_state = _set_random_state(random_state)

    theta = 2 * np.pi * np.sqrt(np.random.rand(num_points//2))

    radius1 = 2 * theta + np.pi
    spiral1 = np.array([np.cos(theta)*radius1, np.sin(theta)*radius1]).T
    noisy_spiral1 = spiral1 + _random_state.normal(size=(num_points//2,2))

    radius2 = -2 * theta - np.pi
    spiral2 = np.array([np.cos(theta)*radius2, np.sin(theta)*radius2]).T
    noisy_sprial2 = spiral2 + _random_state.normal(size=(num_points//2,2)) / 2

    X = np.append(noisy_spiral1, noisy_sprial2, axis=0)
    y = np.append(np.zeros((num_points//2, 1)), np.ones((num_points//2, 1)))

    if dimension > 2:
        noise = _random_state.normal(size=(num_points, dimension - 2)) / np.sqrt(2 * (dimension - 2))
        X = np.append(X, noise, axis=1)

    return _process_data(X, y)


def get_MNIST_data(num_points=None, random_state=None):
    """ Read in csv files from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download. """
    # Best TPC radius = 0.25
    _random_state = _set_random_state(random_state)

    mnist_train = pd.read_csv('MNIST-data/mnist_train.csv')
    X, y = mnist_train.drop(labels='label', axis=1), mnist_train['label']
    factor = 28 * 255 # 28x28 pixel image with grayscale 0-255

    if num_points is not None:
        idx_list = _random_state.choice(range(X.shape[0]), num_points)
        X = X.iloc[idx_list, :]
        y = y.iloc[idx_list]

    return _process_data(X.reset_index(drop=True) / factor, y.reset_index(drop=True))


def get_CIFAR10_data(num_points=None, random_state=None):
    """ Read in file from https://www.cs.toronto.edu/~kriz/cifar.html """
    _random_state = _set_random_state(random_state)

    import pickle
    with open('CIFAR10-data/data_batch_1', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

        factor = 28 * 255 # 28x28 pixel image with grayscale 0-255

        X = dict[b'data']
        y = dict[b'labels']

        if num_points is not None:
            idx_list = _random_state.choice(range(X.shape[0]), num_points)
            X = X[idx_list,:]
            y = [y[idx] for idx in idx_list]

        return _process_data(X / factor, y)



