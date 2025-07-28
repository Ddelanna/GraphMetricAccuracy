import os
os.environ['OMP_NUM_THREADS'] = '4'

from scipy.spatial.distance import squareform, pdist
import pandas as pd
import heapq
import numpy as np
import sklearn
from sklearn.cluster import KMeans


class ProbCoverSampling:
    def __init__(self, unlabeled_points, budget, radius=1.0, distance_matrix=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget
        self.radius = radius

        if distance_matrix is None:
            self.distance_matrix = squareform(pdist(self.unlabeled_points, 'euclidean'))
        else:
            self.distance_matrix = distance_matrix
        self.edges = self._compute_edges()

        self.query_indices = self._get_query_indices()

    def _compute_edges(self):
        """ Create matrix whose (i,j)-entry is 1 if the i-th point and j-th point
        in self.unlabeled_points are within self.radius distance and 0 otherwise. """
        edges = (self.distance_matrix <= self.radius).astype(int)
        return edges

    def __update_edges(self, query_idx):
        """ multiplying the query_idx column by the repulsion factor
            so that edges.sum() has less weight in the vertices that share an edge with the query_idx vertex """
        edges_query_idx = np.where(self.unlabeled_points.index == query_idx)[0][0]  # corresponding index on self.edges
        self.edges[:, np.where(self.edges[edges_query_idx] == 1)] *= 0

    def _get_query_indices(self):
        """ Iteratively picks the point with the highest number of outgoing edges
        (i.e. finds the row in self.edges with the highest number of 1s) """
        query_indices = []

        for _ in range(self.budget):
            # find the index of the row in self.edges with the most 1s
            num_outgoing_edges = self.edges.sum(axis=1) # sum of rows
            most_outgoing_edges, query_idx = heapq.nlargest(1, zip(num_outgoing_edges, self.unlabeled_points.index))[0]

            # check if all data points have been removed from unlabeled pool
            if most_outgoing_edges == 0:
                print('USER WARNING: Under-utilization of budget.')
                break

            # add the index of the data point with the most out-going edges
            query_indices.append(query_idx)

            self.__update_edges(query_idx)

        return np.array(query_indices)


class KMeansSampling:
    def __init__(self, unlabeled_points, budget):
        self.unlabeled_points = unlabeled_points
        self.budget = budget

        self.query_indices = self._get_query_indices()

    def _get_query_indices(self):
        kmeans = sklearn.cluster.KMeans(n_clusters=self.budget,
                                        init='k-means++').fit(self.unlabeled_points)
        centroids = kmeans.cluster_centers_

        dist_matrix = sklearn.metrics.pairwise.pairwise_distances(self.unlabeled_points, centroids)
        query_indices = np.argmin(dist_matrix, axis=0)
        return query_indices


class RandomSampling:
    def __init__(self, unlabeled_points, budget):
        self.unlabeled_points = unlabeled_points

        num_points = self.unlabeled_points.shape[0]
        self.query_indices = np.random.choice(np.arange(0, num_points), budget)


class ConnectedComponentSampling:
    def __init__(self, unlabeled_points, budget, alpha=0.9):
        self.unlabeled_points = unlabeled_points
        self.budget = budget
        self.alpha = alpha

        self._run_algorithm()

    def __get_adjacency_matrix(self, radius):
        """ Returns adjacency matrix with Frobenius distance as the weights between points if their
            Frobenius distance is less than the given radius. Otherewise, they are said to be infinitely far apart. """
        from scipy.sparse.csgraph import dijkstra
        squared_dist_matrix = self.distance_matrix ** 2
        squared_dist_matrix[squared_dist_matrix > radius] = np.inf  # todo : should this be radius**2?
        weighted_adj_matrix = dijkstra(squared_dist_matrix, directed=False)

        return weighted_adj_matrix

    def __find_connected_components(self):
        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(csgraph=self.weighted_adj_matrix, directed=False,
                                                              return_labels=True)
        _, component_sizes = np.unique(component_labels, return_counts=True)

        return n_components, component_labels, component_sizes

    def __sample_from_largest_components(self):
        """ Sample one point from the k largest connected components until (alpha * 100)% of the graph is covered
            or until the connected components are too small """

        component_budgets = [0 for _ in range(self.n_components)]
        coverage = 0

        ordered_components_by_size = sorted(zip(self.component_sizes, np.arange(self.n_components)), reverse=True)
        for component_size, component_idx in ordered_components_by_size:
            # keep going until (coverage is sufficiently large) or until the (clusters are too small) or
            # until (budget is used up)
            if (coverage > self.alpha) or (component_size <= 10) or (sum(component_budgets) == self.budget):
                break
            component_budgets[component_idx] += 1
            coverage += component_size / self.num_points

        return coverage, component_budgets

    def __calculate_coverage(self, radius):
        self.weighted_adj_matrix = self.__get_adjacency_matrix(radius)
        self.n_components, self.component_labels, self.component_sizes = self.__find_connected_components()
        coverage, self.component_budgets = self.__sample_from_largest_components()

        return coverage

    def _find_best_radius(self):
        diameter = np.max(self.distance_matrix)
        tolerance = 0.1 # todo : how to determine tolerance?
        radius_lowerbound, radius_upperbound = 0, diameter
        radius = (radius_upperbound + radius_lowerbound) / 2

        while (radius_upperbound - radius_lowerbound) > tolerance:
            radius = (radius_upperbound + radius_lowerbound) / 2
            coverage = self.__calculate_coverage(radius)
            if coverage < self.alpha: # best radius must be larger than current radius
                # print(f'radius_lowerbound={radius_lowerbound}, r2={radius_upperbound}, coverage={coverage}')
                radius_lowerbound = radius
            else: # best radius must be smaller than current radius
                # print(f'radius_lowerbound={radius_lowerbound}, r2={radius_upperbound}, coverage={coverage}')
                radius_upperbound = radius

        return radius

    def _allot_component_budgets(self):
        # allot budget proportionally to their size
        from math import floor
        remaining_budget = self.budget - sum(self.component_budgets)
        for idx in range(self.n_components):
            self.component_budgets[idx] += floor(self.component_sizes[idx] / self.num_points * remaining_budget)

        # if there is leftover budget, sample randomly proportional to size
        if sum(self.component_budgets) < self.budget:
            distribution = [self.component_sizes[idx] / self.num_points for idx in range(self.n_components)]
            for _ in range(self.budget - sum(self.component_budgets)):
                self.component_budgets[np.random.choice(np.arange(self.n_components), p=distribution)] += 1

    def plot_connected_components(self, data_by_component):
        from matplotlib import pyplot as plt
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, 10))
        fig, ax = plt.subplots()
        color_idx = 0

        for component_label in range(self.n_components):
            if self.component_sizes[component_label] > 10:
                ax.scatter(np.array(data_by_component[component_label])[:, 0],
                           np.array(data_by_component[component_label])[:, 1],
                           color=colors[color_idx], s=1.0)
                color_idx += 1
            else:
                ax.scatter(np.array(data_by_component[component_label])[:, 0],
                           np.array(data_by_component[component_label])[:, 1],
                           color='black', s=1.0)

        fig.show()

    def __split_data_by_components(self):
        data_by_component = []
        for component_label in range(self.n_components):
            data_by_component.append(self.unlabeled_points[self.component_labels == component_label])

        # self.plot_connected_components(data_by_component) # todo : remove?

        return data_by_component

    def __update_query_indices(self, component_data, component_budget):
        # todo : mention the change in distance_matrix input
        if self.query_indices is None:
            self.query_indices = ProbCoverSampling(component_data,
                                                    budget=component_budget,
                                                    radius=self.radius,
                                                    distance_matrix=self.distance_matrix**2).query_indices
        else:
            new_query_indices = ProbCoverSampling(component_data,
                                                   budget=component_budget,
                                                   radius=self.radius,
                                                   distance_matrix=self.distance_matrix**2).query_indices
            self.query_indices = np.append(self.query_indices, new_query_indices)

    def _apply_probcover(self):
        """ Split the data based on connected component and use ProbCover to determine which points to label
            within each component. """

        data_by_component = self.__split_data_by_components()

        self.query_indices = None
        for idx in range(self.n_components):
            if self.component_budgets[idx] != 0:
                self.__update_query_indices(data_by_component[idx], self.component_budgets[idx])

    def _run_algorithm(self):
        from scipy.spatial.distance import squareform, pdist
        self.distance_matrix = squareform(pdist(self.unlabeled_points, 'euclidean'))
        self.num_points = self.unlabeled_points.shape[0]

        self.radius = self._find_best_radius()
        print('\nFinal Radius:', self.radius)

        self._allot_component_budgets()
        self._apply_probcover()

        print("Sizes of components:", self.component_sizes)
        print("Component Budgets:", self.component_budgets)




