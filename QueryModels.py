import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from HelperFunctions import set_random_state


class RandomSampling:
    def __init__(self, unlabeled_points, budget, graph_repr, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget

        _random_state = set_random_state(random_state)
        self.query_indices = _random_state.choice(self.unlabeled_points.index, budget, replace=False)


class KmeansSampling:
    def __init__(self, unlabeled_points, budget, distances, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget

        self._random_state = random_state

        self.query_indices = self._get_query_indices()

    def _get_query_indices(self):
        import sklearn

        kmeans = sklearn.cluster.KMeans(n_clusters=self.budget,
                                        init='k-means++',
                                        random_state=self._random_state).fit(self.unlabeled_points)
        centroids = kmeans.cluster_centers_

        dists_from_centroids = sklearn.metrics.pairwise.pairwise_distances(self.unlabeled_points, centroids)
        query_indices = np.argmin(dists_from_centroids, axis=0)

        return self.unlabeled_points.index[query_indices]


class GraphKmeansSampling:
    def __init__(self, unlabeled_points, budget, distances, random_state=None, n_local_trials=None):
        import graphlearning as gl

        self.unlabeled_points = unlabeled_points
        self.budget = budget
        self.dist_mat = gl.graph(distances)

        self._random_state = set_random_state(random_state)
        if n_local_trials is None:
            self._n_local_trials = 3 + int(np.log(budget))
        else:
            self._n_local_trials = n_local_trials

        self.query_indices = self._get_query_indices()

    def __find_best_sample_point(self, dists_from_query_points):
        rand_vals = self._random_state.uniform(size=self._n_local_trials) * np.sum(dists_from_query_points)
        from sklearn.utils.extmath import stable_cumsum
        candidates_samples_indices = np.searchsorted(stable_cumsum(dists_from_query_points), rand_vals)
        np.clip(candidates_samples_indices, None, dists_from_query_points.size - 1, out=candidates_samples_indices)

        MSE_per_candidate_set = []  # mean squared error per candidate
        dists_per_candidate_set = []
        for candidate_sample_idx in candidates_samples_indices:

            candidate_set_distance = self.dist_mat.dijkstra(bdy_set=[candidate_sample_idx], bdy_val=0) ** 2
            candidate_set_distance = np.minimum(candidate_set_distance, dists_from_query_points)

            MSE_per_candidate_set.append(np.sum(candidate_set_distance))
            dists_per_candidate_set.append(candidate_set_distance)

        min_MSE_idx = np.argmin(MSE_per_candidate_set)
        best_sample_point_idx = candidates_samples_indices[min_MSE_idx]
        dists_from_query_points = dists_per_candidate_set[min_MSE_idx]

        return best_sample_point_idx, dists_from_query_points

    def _get_query_indices(self):
        query_indices = [self._random_state.choice(self.unlabeled_points.shape[0])] # initialize query_indices

        dists_from_query_points = self.dist_mat.dijkstra(bdy_set=query_indices) ** 2
        for _ in range(1, self.budget):
            best_sample_point_idx, dists_from_query_points = self.__find_best_sample_point(dists_from_query_points)
            query_indices.append(best_sample_point_idx)

        return [self.unlabeled_points.index[idx] for idx in query_indices]


class ProbCoverSampling:
    def __init__(self, unlabeled_points, budget, adjacency_matrix, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget
        self.adjacency_matrix = adjacency_matrix
        self._random_state = set_random_state(random_state)

        self.query_indices = self._get_query_indices()

    def __update_edges(self, query_idx):
        """ multiplying the query_idx column by 0 so that edges.sum() has less weight in the
        vertices that share an edge with the query_idx vertex """

        # get the corresponding index on self.adjacency_matrix to the query index on self.unlabeled_points
        adjacency_query_idx = np.where(self.unlabeled_points.index == query_idx)[0][0]

        # get the indices of all the neighboring points to the query point (including the query point itself)
        from scipy.sparse import issparse
        if issparse(self.adjacency_matrix):
            neighboring_point_indices = self.adjacency_matrix.getcol(adjacency_query_idx).indices
        else:
            neighboring_point_indices = np.where(self.adjacency_matrix[adjacency_query_idx] == 1)

        # remove all outgoing edges of the neighboring points
        self.adjacency_matrix[:, neighboring_point_indices] *= 0

    def _get_query_indices(self):
        """ Iteratively picks the point with the highest number of outgoing edges
        (i.e. finds the row in self.edges with the highest number of 1s) """
        query_indices = []

        for _ in range(self.budget):
            from heapq import nlargest
            # find the index of the row in self.edges with the most 1s
            num_outgoing_edges = self.adjacency_matrix.sum(axis=1) # sum of rows
            most_outgoing_edges, query_idx = nlargest(1, zip(num_outgoing_edges, self.unlabeled_points.index))[0]

            # check if all data points have been removed from unlabeled pool
            if most_outgoing_edges == 0:
                try:
                    random_query_indices = self._random_state.choice(np.setdiff1d(self.unlabeled_points.index, query_indices),
                                                          size=self.budget-len(query_indices),
                                                          replace=False)
                    query_indices = np.concatenate((query_indices, random_query_indices))
                    return query_indices
                except ValueError:
                    raise f'USER WARNING: Under-utilization of budget.'

            # add the index of the data point with the most out-going edges
            query_indices.append(query_idx)

            self.__update_edges(query_idx)

        return query_indices

