import numpy as np
from HelperFunctions import set_random_state
import graphlearning as gl
import pandas as pd



def _process_data(data, labels, num_points, random_state):
    """ Ensures that the data and labels are of the correct type and have the same indexing to
        make querying easier later on. """
    _random_state = set_random_state(random_state)

    labels = pd.Series(labels, index=np.arange(data.shape[0]))

    data = pd.DataFrame(data, index=np.arange(data.shape[0]))
    data['query index'] = labels.index
    data.set_index('query index', inplace=True)

    if num_points is not None:
        idx_list = _random_state.choice(range(data.shape[0]), num_points, replace=False)
        data = data.iloc[idx_list, :]
        labels = labels.iloc[idx_list]

    return data, labels


def create_spiral_data(num_points, dimension=2, random_state=None):
    _random_state = set_random_state(random_state)

    theta = 2 * np.pi * np.sqrt(np.random.rand(num_points // 2))

    radius1 = 2 * theta + np.pi
    spiral1 = np.array([np.cos(theta) * radius1, np.sin(theta) * radius1]).T
    noisy_spiral1 = spiral1 + _random_state.normal(size=(num_points // 2, 2))

    radius2 = -2 * theta - np.pi
    spiral2 = np.array([np.cos(theta) * radius2, np.sin(theta) * radius2]).T
    noisy_sprial2 = spiral2 + _random_state.normal(size=(num_points // 2, 2)) / 2

    X = np.append(noisy_spiral1, noisy_sprial2, axis=0)
    y = np.append(np.zeros((num_points // 2, 1)), np.ones((num_points // 2, 1)))

    if dimension > 2:
        noise = _random_state.normal(size=(num_points, dimension - 2)) / np.sqrt(2 * (dimension - 2))
        X = np.append(X, noise, axis=1)

    return _process_data(X, y, None, random_state)


def get_MNIST_data(num_points=None, random_state=None):
    """ Read in csv files from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download. """

    # mnist_train = pd.read_csv('data/MNIST-data/mnist_train.csv')
    # X, y = mnist_train.drop(labels='label', axis=1), mnist_train['label']
    X, y = gl.datasets.load(dataset="mnist", metric="vae")
    factor = 28 * 255  # 28x28 pixel image with grayscale 0-255

    return _process_data(X / factor, y, num_points, random_state)


def get_CIFAR10_data(num_points=None, random_state=None):
    X, y = gl.datasets.load('cifar', metric='aet', labels_only=False)
    return _process_data(X, y, num_points, random_state)


def get_FASHIONMNIST_data(num_points=None, random_state=None):
    X, y = gl.datasets.load('fashionmnist', metric='vae', labels_only=False)
    return _process_data(X, y, num_points, random_state)


def get_Satellite_data(num_points, random_state=None):
    X = np.loadtxt('data/data_satellite.txt', usecols=range(36))
    y = np.loadtxt('data/ground_truth_satellite.txt', usecols=range(1))
    y = y.astype(int)

    return _process_data(X, y, num_points, random_state)


def get_USPS_data(num_points, random_state=None):
    X = np.loadtxt('data/data_usps.txt', usecols=range(256))
    y = np.loadtxt('data/ground_truth_usps.txt', usecols=range(1))
    y = y.astype(int)

    return _process_data(X, y, num_points, random_state)


def get_COIL20_data(num_points, random_state=None):
    X = np.loadtxt('data/data_coil20.txt', usecols=range(1024))
    y = np.loadtxt('data/ground_truth_coil20.txt', usecols=range(1))
    y = y.astype(int)

    return _process_data(X, y, num_points, random_state)


def get_OPTDIGITS_data(num_points, random_state=None):
    X = np.loadtxt('data/data_optdigits.txt', usecols=range(64))
    y = np.loadtxt('data/ground_truth_optdigits.txt', usecols=range(1))
    y = y.astype(int)

    return _process_data(X, y, num_points, random_state)


def get_multmoons_data(num_points=None, random_state=None):
    from sklearn.datasets import make_moons
    data1, labels1 = make_moons(n_samples=1000, noise=0.15, random_state=0)
    data2, labels2 = make_moons(n_samples=1000, noise=0.15, random_state=1)
    data3, labels3 = make_moons(n_samples=1000, noise=0.15, random_state=2)
    data2 += np.array([4, 0.0])  # shift second moons
    data3 += np.array([8, 0.0])  # shift third moons
    data = np.vstack([data1, data2, data3])
    labels = np.hstack([labels1, labels2 + 2, labels3 + 4])  # different labels for each moon
    even = labels % 2 == 0
    data[even] += np.array([-0.2, 0.2])  # shift even moons up slightly to avoid overlap
    data[~even] += np.array([0.2, -0.2])  # shift odd moons down slightly to avoid overlap
    labels = 1 * (~even)

    return _process_data(data, labels, num_points, random_state)


def get_swissroll_data(num_points=None, random_state=None):
    from sklearn.datasets import make_swiss_roll
    data1, labels1 = make_swiss_roll(n_samples=1500, noise=0.5, random_state=0)
    data2, labels2 = make_swiss_roll(n_samples=1500, noise=0.5, random_state=1)
    data2 = data2 @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # rotate to intertwine

    X = np.vstack([data1, data2])
    y = np.concatenate((np.zeros(data1.shape[0]), np.ones(data2.shape[0])))

    return _process_data(X, y, num_points, random_state)


def get_smileyface_data(num_points=None, random_state=None):
    d = 10  # other dimension for the eyes
    n_head = 800
    n_mouth = 200
    n_eye = 400
    rand_state = np.random.RandomState(0)

    r_head = 3
    r_mouth = 1.5
    head_std = 0.01
    mouth_std = 0.05
    eye_std = 0.05

    # head
    head = rand_state.randn(n_head, 2)
    head /= np.linalg.norm(head, axis=1, keepdims=True) / r_head
    head += head_std * rand_state.randn(n_head, 2)

    # mouth
    mouth = np.linspace(-np.pi / 4.0, -3 * np.pi / 4.0, n_mouth)
    mouth = np.column_stack([np.cos(mouth), np.sin(mouth)]) * r_mouth
    mouth += mouth_std * rand_state.randn(n_mouth, 2)

    # eyes
    eye = eye_std * rand_state.randn(n_eye, d)
    right_eye = eye + np.array([[1, 1] + [0] * (d - 2)])
    left_eye = eye + np.array([[-1, 1] + [0] * (d - 2)])

    X = np.vstack(
        [np.hstack((head, np.zeros((n_head, d - 2)))), np.hstack((mouth, np.zeros((n_mouth, d - 2)))), right_eye,
         left_eye])
    y = np.hstack(
        [np.zeros(n_head), np.zeros(n_mouth), np.ones(n_eye), np.ones(n_eye)])  # label eyes as 1, rest as 0

    return _process_data(X, y, num_points, random_state)



