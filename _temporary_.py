import os

os.environ['OMP_NUM_THREADS'] = '4'

import heapq
import numpy as np
from HelperFunctions import AdjacencyMatrices, _set_random_state
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import graphlearning as gl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import scipy.io as sio
import os
from sklearn.cluster import kmeans_plusplus
import PredictionModels
import matplotlib.pyplot as plt


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Warns if the final cumulative sum does not match the sum (up to the chosen
    tolerance).

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.

    Returns
    -------
    out : ndarray
        Array with the cumulative sums along the chosen axis.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.allclose(
            out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
    ):
        warnings.warn(
            (
                "cumsum was found to be unstable: "
                "its last element does not correspond to sum"
            ),
            RuntimeWarning,
        )
    return out


def kmeansplusplus(X, labels, distance_matrix, budget, n_local_trials=None, random_state=None):
    centers, indices = kmeans_plusplus(X, n_clusters=budget, n_local_trials=n_local_trials, random_state=random_state)
    return centers, indices


def adaptive_sampling(X, labels, distance_matrix, budget, distance_type='Fermat', n_local_trials=None,
                      random_state=None):
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    sample_weight = np.ones(X.shape[0])
    n_clusters = budget
    n_samples, n_features = X.shape
    if n_local_trials is None:
        n_local_trials = 3 + int(np.log(n_clusters))
    # print(X.shape)
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    indices = np.full(n_clusters, -1, dtype=int)

    # center_ids = gl.trainsets.generate(labels, 1, seed=42)

    # center_ids = random_state.integers(n_samples)
    center_id = random_state.integers(n_samples)
    # center_id = random_state.randint(n_samples)
    print(center_id)

    W = distance_matrix
    coreset = [center_id]  # initialize with first center

    if sp.issparse(X):
        centers[0] = X[[center_id]].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    def plot_prop(prop_vals, show_source=True):
        # assert prop_vals.size == n
        fig, ax = plt.subplots()
        p = ax.scatter(X[:, 0], X[:, 1], c=prop_vals)
        if show_source:
            ax.scatter(X[coreset, 0], X[coreset, 1], c='r', s=80, marker='^')
        plt.colorbar(p, ax=ax)
        plt.show()

    def compute_distance(W, bdy_set):
        if distance_type == 'dijkstra':
            # W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
            G = gl.graph(W)
            dist = G.dijkstra(bdy_set=bdy_set, bdy_val=0)
            dists = dist

            return dists

        elif distance_type == 'peikonal':
            # W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
            G = gl.graph(W)
            dist = G.peikonal(bdy_set=bdy_set, p=1)
            dists = dist ** 2
            return dists

        elif distance_type == 'Fermat':
            p = 2
            # W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
            W_powered = W.copy()
            W_powered **= p
            W = W_powered
            G = gl.graph(W)
            dist = G.dijkstra(bdy_set=bdy_set, bdy_val=0)
            dists = dist ** 2
            return dists

        elif distance_type == 'Euclidean':
            p = 1
            # W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
            W_powered = W.copy()
            W_powered.data **= p
            W = W_powered
            G = gl.graph(W)
            dist = G.dijkstra(bdy_set=bdy_set, bdy_val=0)
            dists = dist ** 2
            return dists

        elif distance_type == 'poisson_propagation':
            # W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
            G = gl.graph(W)
            poiss_prop = poisson_propagation(G, idx=bdy_set, tau=1e-6)
            dists = 1. / (1e-1 + poiss_prop)

            dists = dists - dists.min()
            dists = dists ** 2
            return dists
        else:
            raise ValueError(f"Unsupported distance method: {distance_type}")

    dists = compute_distance(W, coreset)

    current_pot = np.sum(dists)

    for c in range(1, n_clusters):

        # plot_prop(dists)

        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidates_samples = np.searchsorted(stable_cumsum(dists), rand_vals)
        # print(candidates_samples)

        np.clip(candidates_samples, None, dists.size - 1, out=candidates_samples)

        # candidates_samples = np.random.choice(n_samples, n_local_trials, p=probs, replace=False)

        sums = []
        dists_per_candidate = []
        for s in candidates_samples:
            bdy_set = indices[:c].tolist() + [s]  # Combine candidate and previous centers

            d = compute_distance(W, bdy_set)

            sums.append(np.sum(d))
            dists_per_candidate.append(d)
        min_index = np.argmin(sums)
        best_sample_point = candidates_samples[min_index]
        dists = dists_per_candidate[min_index]

        current_pot = np.sum(dists)

        if sp.issparse(X):
            centers[c] = X[[best_sample_point]].toarray()
        else:
            centers[c] = X[best_sample_point]
        indices[c] = best_sample_point

    return centers, indices


def adaptive(X, data, distance_matrix, labels, budget=None, distance_type='Fermat', model_type='1-NN'):
    NUM_SEEDS = 10
    y = len(np.unique(labels))  # number of classes

    ACC_matrix = np.zeros((NUM_SEEDS, budget - y + 1))  # +1 because of y-1 offset

    for seed_idx, seed in enumerate([42, 51, 61, 71, 81, 91, 101, 111, 121, 131]):
        # acc = np.zeros(num_choices + 1)  # Ensure correct length
        acc = []
        list_num_labels = []

        if distance_type == 'Fermat':
            centers, train_indices = adaptive_sampling(data, labels, distance_matrix, budget=budget,
                                                       distance_type=distance_type, n_local_trials=10,
                                                       random_state=seed)
        elif distance_type == 'peikonal':
            centers, train_indices = adaptive_sampling(data, labels, distance_matrix, budget=budget,
                                                       distance_type=distance_type, n_local_trials=10,
                                                       random_state=seed)
        elif distance_type == 'dijkstra':
            centers, train_indices = adaptive_sampling(data, labels, distance_matrix, budget=budget,
                                                       distance_type=distance_type, n_local_trials=10,
                                                       random_state=seed)
        elif distance_type == 'Euclidean':
            centers, train_indices = kmeansplusplus(data, labels, distance_matrix, budget=budget, n_local_trials=10,
                                                    random_state=seed)

        for i, idx in enumerate(range(y - 1, budget)):

            # current_train_indices = np.array(train_indices[:idx+1])

            if model_type == '1-NN':
                model = KNeighborsClassifier(n_neighbors=1)
                current_train_indices = np.array(train_indices[:idx + 1], dtype=int).ravel()
                train_labels = labels[current_train_indices]
                test_indices = np.setdiff1d(np.arange(len(data)), current_train_indices)
                X_train, y_train = data[current_train_indices], labels[current_train_indices]
                X_test, y_test = data[test_indices], labels[test_indices]
                # print(train_labels)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred) * 100
                acc.append(accuracy)

                # acc[i] = accuracy
                print(f"Seed {seed}, #labels: {len(current_train_indices)}, accuracy: {accuracy:.4f}")
                list_num_labels.append(len(current_train_indices))


            elif model_type == 'GMA':
                current_train_indices = np.array(train_indices[:idx + 1], dtype=int).ravel()
                score = PredictionMethods.GraphMetricAccuracy(data, current_train_indices, labels, 0.08).score
                accuracy = round(score * 100, 2)
                print(accuracy)
                acc.append(accuracy)
                print(f"Seed {seed}, #labels: {len(current_train_indices)}, accuracy: {accuracy:.4f}")
                list_num_labels.append(len(current_train_indices))

        ACC_matrix[seed_idx, :] = acc

    avg_accuracy = ACC_matrix.mean(axis=0)
    print(avg_accuracy)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_path = os.path.join(results_dir, f'adaptive_{X}_{distance_type}_{model_type}.mat')

    sio.savemat(file_path, {
        'avg_accuracy': avg_accuracy,
        'ACC_matrix': ACC_matrix,
        'list_num_labels': np.array(list_num_labels)
    })

    return avg_accuracy, list_num_labels

