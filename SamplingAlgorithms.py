import os
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
from HelperFunctions import _set_random_state


class RandomSampling:
    def __init__(self, unlabeled_points, budget, graph_repr=None, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget

        _random_state = _set_random_state(random_state)
        self.query_indices = _random_state.choice(self.unlabeled_points.index, budget, replace=False)


class KmeansSampling:
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
            MSE_per_candidate.append(np.sum(candidate_distance))
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
        num_points = self.unlabeled_points.shape[0]

        # ensure all large components have budget >= 1
        component_budgets = self.__sample_from_largest_components()

        # allot budget proportionally to component size
        from math import floor
        remaining_budget = self.budget - sum(component_budgets)
        for idx in range(self.n_components):
            component_budgets[idx] += floor(self.component_sizes[idx] / num_points * remaining_budget)

        # if there is leftover budget, sample randomly proportional to size
        if sum(component_budgets) < self.budget:
            distribution = [self.component_sizes[idx] / num_points for idx in range(self.n_components)]
            for _ in range(self.budget - sum(component_budgets)):
                component_budgets[self._random_state.choice(np.arange(self.n_components), p=distribution)] += 1

        return component_budgets

    def __update_query_indices(self, query_indices, component_data_indices, component_budget):
        sub_adjacency_matrix = self.adjacency_matrix[np.ix_(component_data_indices, component_data_indices)]

        if query_indices is None:
            query_indices = ProbCoverSampling(self.unlabeled_points.iloc[component_data_indices],
                                              budget=component_budget,
                                              adjacency_matrix=sub_adjacency_matrix).query_indices
        else:
            new_query_indices = ProbCoverSampling(self.unlabeled_points.iloc[component_data_indices],
                                                  budget=component_budget,
                                                  adjacency_matrix=sub_adjacency_matrix).query_indices
            query_indices = np.append(query_indices, new_query_indices)

        return query_indices

    def _apply_probcover(self):
        """ Split the data based on connected component and use ProbCover to determine which points to label
            within each component. """

        query_indices = None
        for label in range(self.n_components):
            if self.component_budgets[label] != 0:
                component_data_indices = np.where(self.component_labels == label)[0]
                query_indices = self.__update_query_indices(query_indices, component_data_indices, self.component_budgets[label])

        return query_indices

    def _run_algorithm(self):
        self.n_components, self.component_labels, self.component_sizes = self._find_connected_components()
        self.component_budgets = self._allot_component_budgets()
        self.query_indices = self._apply_probcover()







