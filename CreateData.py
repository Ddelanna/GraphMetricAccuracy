import numpy as np
import pandas as pd


def _process_data(data, labels):
    """ Ensures that the data and labels are of the correct type and have the same indexing to
        make querying easier later on. """
    labels = pd.Series(labels, index=np.arange(data.shape[0]))

    data = pd.DataFrame(data, index=np.arange(data.shape[0]))
    data['query index'] = labels.index
    data.set_index('query index', inplace=True)

    return data, labels


def create_spiral_data(num_points, dimension=2):
    theta = 2 * np.pi * np.sqrt(np.random.rand(num_points//2))

    radius1 = 2 * theta + np.pi
    data1 = np.array([np.cos(theta)*radius1, np.sin(theta)*radius1]).T
    noisy_data1 = data1 + np.random.normal(size=(num_points//2,2))

    radius2 = -2 * theta - np.pi
    data2 = np.array([np.cos(theta)*radius2, np.sin(theta)*radius2]).T
    noisy_data2 = data2 + np.random.normal(size=(num_points//2,2))

    X = np.append(noisy_data1, noisy_data2, axis=0)
    y = np.append(np.zeros((num_points//2, 1)), np.ones((num_points//2, 1)))

    if dimension > 2:
        noise = np.random.normal(size=(num_points, dimension - 2)) / np.sqrt(dimension - 2)
        X = np.append(X, noise, axis=1)

    return _process_data(X, y)



