import os
os.environ['OMP_NUM_THREADS'] = '4'

import heapq
import numpy as np
from HelperFunctions import AdjacencyMatrices, _set_random_state



class KmeansSampling:
    def __init__(self, unlabeled_points, budget, distance_matrix, distance_type='peikonal',
                 n_local_trials=None, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget

        if n_local_trials is None:
            self._n_local_trials = 3 + int(np.log(budget))
        else:
            self._n_local_trials = n_local_trials

        self._random_state = _set_random_state(random_state)
        self.distance_matrix = distance_matrix
        self._distance_type = distance_type

        self.query_indices = self._get_query_indices()

    def __find_best_sample_point(self, query_indices, dists):
        rand_vals = self._random_state.uniform(size=self._n_local_trials) * np.sum(dists)
        from sklearn.utils.extmath import stable_cumsum
        candidates_samples = np.searchsorted(stable_cumsum(dists), rand_vals)
        np.clip(candidates_samples, None, dists.size - 1, out=candidates_samples)

        MSE_per_candidate =[]  # mean squared error
        dists_per_candidate = []
        for candidate_sample in candidates_samples:
            boundary_set = query_indices + [candidate_sample] # combine candidate and previous centers
            candidate_distance = AdjacencyMatrices().compute_distance(self.distance_matrix, boundary_set,
                                                                      distance_type=self._distance_type)
            MSE_per_candidate.append(np.sum(candidate_distance))
            dists_per_candidate.append(candidate_distance)
        min_MSE_idx = np.argmin(MSE_per_candidate)
        best_sample_point = candidates_samples[min_MSE_idx]
        dists = dists_per_candidate[min_MSE_idx] ** 2

        return best_sample_point, dists

    def _get_query_indices(self):
        query_indices = [self._random_state.choice(self.unlabeled_points.index)] # initialize query_indices

        dists = AdjacencyMatrices().compute_distance(self.distance_matrix, query_indices, distance_type=self._distance_type) ** 2
        for idx in range(1, self.budget):
            best_sample_point, dists = self.__find_best_sample_point(query_indices, dists)
            query_indices.append(best_sample_point)

        return query_indices


class RandomSampling:
    def __init__(self, unlabeled_points, budget, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget

        _random_state = _set_random_state(random_state)
        self.query_indices = _random_state.choice(self.unlabeled_points.index, budget)


class ProbCoverSampling:
    def __init__(self, unlabeled_points, budget, adjacency_matrix=None, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget
        self.adjacency_matrix = adjacency_matrix
        self._random_state = _set_random_state(random_state)

        self.query_indices = self._get_query_indices()

    def __update_edges(self, query_idx):
        """ multiplying the query_idx column by 0 so that edges.sum() has less weight in the
        vertices that share an edge with the query_idx vertex """
        # get the corresponding index on self.adjacency_matrix to the query index on self.unlabeled_points
        adjacency_query_idx = np.where(self.unlabeled_points.index == query_idx)[0][0]

        # remove all outgoing edges of the labeled point AND all of its surrounding points
        from scipy.sparse import issparse
        if issparse(self.adjacency_matrix):
            neighboring_point_indices = self.adjacency_matrix.getcol(adjacency_query_idx).indices
            self.adjacency_matrix[:, neighboring_point_indices] *= 0
        else:
            neighboring_point_indices = np.where(self.adjacency_matrix[adjacency_query_idx] == 1)
            self.adjacency_matrix[:, neighboring_point_indices] *= 0

    def _get_query_indices(self):
        """ Iteratively picks the point with the highest number of outgoing edges
        (i.e. finds the row in self.edges with the highest number of 1s) """
        query_indices = []

        for _ in range(self.budget):
            # find the index of the row in self.edges with the most 1s
            num_outgoing_edges = self.adjacency_matrix.sum(axis=1) # sum of rows
            most_outgoing_edges, query_idx = heapq.nlargest(1, zip(num_outgoing_edges, self.unlabeled_points.index))[0]

            # check if all data points have been removed from unlabeled pool
            if most_outgoing_edges == 0:
                print('USER WARNING: Under-utilization of budget.')
                query_idx = self._random_state.choice(np.setdiff1d(self.unlabeled_points.index, query_indices))

            # add the index of the data point with the most out-going edges
            query_indices.append(query_idx)

            self.__update_edges(query_idx)

        return query_indices


class ConnectedComponentSampling:
    def __init__(self, unlabeled_points, budget, adjacency_matrix=None, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget
        self.adjacency_matrix = adjacency_matrix

        self._random_state = _set_random_state(random_state)

        self._run_algorithm()

    def _find_connected_components(self):
        """ :return n_components: number of connected components of the graph
            :return component_labels: corresponding connected component label of each data point
            :return component_sizes: number of data points in each connected component in order of their component label """

        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(csgraph=self.adjacency_matrix, directed=False,
                                                              return_labels=True)
        _, component_sizes = np.unique(component_labels, return_counts=True)

        return n_components, component_labels, component_sizes

    def __sample_from_largest_components(self):
        """ While we are under budget, sample one point from the k largest connected components
            until the connected components are too small """

        component_budgets = [0 for _ in range(self.n_components)]

        ordered_components_by_size = sorted(zip(self.component_sizes, np.arange(self.n_components)), reverse=True)
        for component_size, component_idx in ordered_components_by_size:
            # keep going until the (clusters are too small) or until (budget is used up)
            if (component_size <= 10) or (sum(component_budgets) == self.budget):
                break
            component_budgets[component_idx] += 1

        return component_budgets

    def _allot_component_budgets(self):
        self.num_points = self.unlabeled_points.shape[0]

        # ensure all large components have budget >= 1
        component_budgets = self.__sample_from_largest_components()

        # allot budget proportionally to component size
        from math import floor
        remaining_budget = self.budget - sum(component_budgets)
        for idx in range(self.n_components):
            component_budgets[idx] += floor(self.component_sizes[idx] / self.num_points * remaining_budget)

        # if there is leftover budget, sample randomly proportional to size
        if sum(component_budgets) < self.budget:
            distribution = [self.component_sizes[idx] / self.num_points for idx in range(self.n_components)]
            for _ in range(self.budget - sum(component_budgets)):
                component_budgets[self._random_state.choice(np.arange(self.n_components), p=distribution)] += 1

        return component_budgets

    def __update_query_indices(self, query_indices, component_data, component_budget):
        sub_adjacency_matrix = self.adjacency_matrix[np.ix_(component_data.index, component_data.index)]

        if query_indices is None:
            query_indices = ProbCoverSampling(component_data,
                                              budget=component_budget,
                                              adjacency_matrix=sub_adjacency_matrix,
                                              random_state=self._random_state).query_indices
        else:
            new_query_indices = ProbCoverSampling(component_data,
                                                  budget=component_budget,
                                                  adjacency_matrix=sub_adjacency_matrix,
                                                  random_state=self._random_state).query_indices
            query_indices = np.append(query_indices, new_query_indices)

        return query_indices

    def _apply_probcover(self):
        """ Split the data based on connected component and use ProbCover to determine which points to label
            within each component. """

        data_by_component = [self.unlabeled_points[self.component_labels == label] for label in range(self.n_components)]

        query_indices = None
        for idx in range(self.n_components):
            if self.component_budgets[idx] != 0:
                query_indices = self.__update_query_indices(query_indices, data_by_component[idx], self.component_budgets[idx])

        return query_indices

    def _run_algorithm(self):
        self.n_components, self.component_labels, self.component_sizes = self._find_connected_components()
        self.component_budgets = self._allot_component_budgets()
        self.query_indices = self._apply_probcover()



