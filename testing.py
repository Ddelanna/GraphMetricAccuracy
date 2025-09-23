import os
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import sklearn
from HelperFunctions import _set_random_state
from CreateData import create_spiral_data
from HelperFunctions import AdjacencyMatrices
from time import time
import graphlearning as gl


class OldKmeans:
    def __init__(self, unlabeled_points, budget, graph_repr, random_state=None, n_local_trials=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget
        self.graph = graph_repr

        self._random_state = _set_random_state(random_state)
        if n_local_trials is None:
            self._n_local_trials = 3 + int(np.log(budget))
        else:
            self._n_local_trials = n_local_trials

        self.query_indices = self._get_query_indices()

    def __find_best_sample_point(self, query_indices, dists):
        rand_vals = self._random_state.uniform(size=self._n_local_trials) * np.sum(dists)
        from sklearn.utils.extmath import stable_cumsum
        candidates_samples_indices = np.searchsorted(stable_cumsum(dists), rand_vals)
        np.clip(candidates_samples_indices, None, dists.size-1, out=candidates_samples_indices)

        MSE_per_candidate = []  # mean squared error per candidate
        dists_per_candidate = []
        for candidate_sample_idx in candidates_samples_indices:
            candidate_distance = self.graph.dijkstra(bdy_set=query_indices + [candidate_sample_idx], bdy_val=0)
            MSE_per_candidate.append(np.sum(candidate_distance)) # todo: should this be squared?
            dists_per_candidate.append(candidate_distance)
        min_MSE_idx = np.argmin(MSE_per_candidate)
        best_sample_point = candidates_samples_indices[min_MSE_idx]
        dists = dists_per_candidate[min_MSE_idx] ** 2

        return best_sample_point, dists

    def _get_query_indices(self):
        query_indices = [self._random_state.choice(self.unlabeled_points.shape[0])] # initialize query_indices

        dists = self.graph.dijkstra(bdy_set=query_indices)
        for _ in range(1, self.budget):
            best_sample_point_idx, dists = self.__find_best_sample_point(query_indices, dists)
            query_indices.append(best_sample_point_idx)

        return [self.unlabeled_points.index[idx] for idx in query_indices]

class SklearnKmeans:
    def __init__(self, unlabeled_points, budget, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget

        self._random_state = random_state

        self.query_indices = self._get_query_indices()

    def _get_query_indices(self):
        kmeans = sklearn.cluster.KMeans(n_clusters=self.budget,
                                        init='k-means++',
                                        random_state=self._random_state).fit(self.unlabeled_points)
        centroids = kmeans.cluster_centers_

        dists_from_centroids = sklearn.metrics.pairwise.pairwise_distances(self.unlabeled_points, centroids)
        query_indices = np.argmin(dists_from_centroids, axis=0)
        return query_indices


class NewKmeans:
    def __init__(self, unlabeled_points, budget, graph_repr, random_state=None, n_local_trials=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget
        self.graph = graph_repr

        self._random_state = _set_random_state(random_state)
        if n_local_trials is None:
            self._n_local_trials = 3 + int(np.log(budget))
        else:
            self._n_local_trials = n_local_trials

        self.query_indices = self._get_query_indices()

    def __find_best_sample_point(self, query_indices, dists):
        rand_vals = self._random_state.uniform(size=self._n_local_trials) * np.sum(dists)
        from sklearn.utils.extmath import stable_cumsum
        candidates_samples_indices = np.searchsorted(stable_cumsum(dists), rand_vals)
        np.clip(candidates_samples_indices, None, dists.size-1, out=candidates_samples_indices)

        from scipy.sparse.csgraph import dijkstra
        MSE_per_candidate = []  # mean squared error per candidate
        dists_per_candidate = []
        print('NEW ROUND')
        for candidate_sample_idx in candidates_samples_indices:
            candidate_distance = dijkstra(self.graph, indices=query_indices + [candidate_sample_idx], directed=False)
            # print('\nexact', candidate_distance)
            candidate_distance = np.min(candidate_distance, axis=0)

            candidate_distance = dijkstra(self.graph, indices=candidate_sample_idx, directed=False)
            # print('dists', dists)
            # print('candidate_distance', candidate_distance)
            candidate_distance = np.minimum(dists, candidate_distance)


            MSE_per_candidate.append(np.sum(candidate_distance)) # todo: should this be squared?
            dists_per_candidate.append(candidate_distance)
        min_MSE_idx = np.argmin(MSE_per_candidate)
        best_sample_point_idx = candidates_samples_indices[min_MSE_idx]
        dists = dists_per_candidate[min_MSE_idx]

        return best_sample_point_idx, dists

    def _get_query_indices(self):
        query_indices = [self._random_state.choice(self.unlabeled_points.shape[0])] # initialize query_indices

        from scipy.sparse.csgraph import dijkstra
        dists = dijkstra(self.graph, indices=query_indices, directed=False)
        for _ in range(1, self.budget):
            best_sample_point_idx, dists = self.__find_best_sample_point(query_indices, dists)
            query_indices.append(best_sample_point_idx)

        return [self.unlabeled_points.index[idx] for idx in query_indices]


if __name__ == '__main__':
    data, oracle = create_spiral_data(1000, random_state=50)

    time1 = time()
    dist = AdjacencyMatrices().full_graph(data, metric='euclidean')
    time2 = time()
    KM = OldKmeans(data, 10, dist, random_state=1).query_indices
    print('OLD', time()-time2, time2-time1, KM)

    # time1 = time()
    # graph_repr = AdjacencyMatrices().distance_matrix(data, metric='1fermat')
    # time2 = time()
    # KMT = NewKmeans(data, 10, graph_repr, random_state=1).query_indices
    # print(time()-time2, time2-time1, KMT)

    time1 = time()
    graph_repr = AdjacencyMatrices().distance_matrix(data, metric='euclidean')
    time2 = time()
    KMT = NewKmeans(data, 10, graph_repr, random_state=1).query_indices
    print('NEW', time() - time2, time2 - time1, KMT)

    time1 = time()
    KMT = SklearnKmeans(data, 10).query_indices
    print(time() - time1)
