import matplotlib.pyplot as plt
import numpy as np
import sklearn
from math import floor


def set_random_state(random_state):
    if isinstance(random_state, int):
        return np.random.default_rng(random_state)
    return np.random.default_rng(np.random.random_integers(low=0, high=10000))


class Plots:
    @staticmethod
    def plot_confusion_matrix(cm, title=''):
        cm_display = sklearn.metrics.ConfusionMatrixDisplay(cm)
        cm_display.plot()
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_avg_scores(x, y_avg, y_std=None, title=''):
        fig, ax = plt.subplots(1)
        ax.plot(x, y_avg)
        if y_std is not None:
            ax.fill_between(x, y_avg + y_std, y_avg + (-1)*y_std,
                            facecolor='blue', alpha=0.25)
        plt.title(title)
        plt.show()


class AdjacencyMatrices:
    @staticmethod
    def _check_sparsity(matrix, sparse):
        if not sparse:
            return matrix
        from scipy.sparse import csc_matrix
        return csc_matrix(matrix)

    @staticmethod
    def distance_matrix(data, metric='euclidean'):
        from scipy.spatial.distance import squareform, pdist

        if (metric == 'euclidean') or ('1' in metric):
            return squareform(pdist(data, metric='euclidean'))
        elif (metric == 'sqeuclidean') or ('2' in metric):
            return squareform(pdist(data, metric='sqeuclidean'))
        else:
            raise ValueError('Metric must be either \'euclidean\' or \'pfermat\'.')

    def knn_graph(self, data, k=1, radius=1.0, metric='euclidean', sparse=True):
        from sklearn.neighbors import kneighbors_graph
        from scipy.sparse.csgraph import dijkstra

        if metric == 'euclidean':
            knn_graph = kneighbors_graph(data, n_neighbors=k, metric='euclidean', mode='connectivity').toarray()
            knn_graph = np.maximum(knn_graph, knn_graph.transpose())
        elif ('fermat' in metric) and ('1' in metric):
            knn_graph = kneighbors_graph(data, n_neighbors=k, metric='euclidean', mode='distance').toarray()
            knn_graph [knn_graph  == 0] = np.inf
            np.fill_diagonal(knn_graph, 0)
            knn_graph = np.minimum(knn_graph, knn_graph.transpose())
            graph_distance_matrix = dijkstra(knn_graph, directed=False)
            knn_graph = (graph_distance_matrix <= 2 * radius).astype(int)
        elif ('fermat' in metric) and ('2' in metric):
            knn_graph = kneighbors_graph(data, n_neighbors=k, metric='sqeuclidean', mode='distance').toarray()
            knn_graph[knn_graph == 0] = np.inf
            np.fill_diagonal(knn_graph, 0)
            knn_graph = np.minimum(knn_graph, knn_graph.transpose())
            graph_distance_matrix = dijkstra(knn_graph, directed=False)
            knn_graph = (graph_distance_matrix <= 2 * radius**2).astype(int)
        else:
            raise ValueError('Metric must be either \'euclidean\' or \'pfermat\'.')

        return self._check_sparsity(knn_graph, sparse)

    def epsilon_graph(self, data, radius=1.0, metric='euclidean', sparse=True):
        from scipy.sparse.csgraph import dijkstra
        distance_matrix = self.distance_matrix(data, metric='euclidean')

        if metric == 'euclidean':
            epsilon_graph = (distance_matrix <= radius).astype(int)
        elif ('fermat' in metric) and ('1' in metric):
            distance_matrix[distance_matrix > radius] = np.inf
            graph_distance_matrix = dijkstra(distance_matrix, directed=False)
            epsilon_graph = (graph_distance_matrix <= 2 * radius).astype(int)
        elif ('fermat' in metric) and ('2' in metric):
            distance_matrix[distance_matrix > radius] = np.inf
            graph_distance_matrix = dijkstra(distance_matrix, directed=False)
            epsilon_graph = (graph_distance_matrix <= 2 * radius**2).astype(int)
        else:
            raise ValueError('Metric must be either \'euclidean\' or \'pfermat\'.')

        return self._check_sparsity(epsilon_graph, sparse)


class BestParameter:
    def __init__(self, data):
        self.data = data

    def best_radius(self, alpha):
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage
        distance_matrix = pdist(self.data) # todo: sparse matrix
        agglomerative_clustering = linkage(distance_matrix, 'single')

        cluster_sizes = {i: 1 for i in range(self.data.shape[0])}
        for step, (i, j, distance, _) in enumerate(agglomerative_clustering):
            new_cluster_id = self.data.shape[0] + step
            cluster_sizes[new_cluster_id] = cluster_sizes[i] + cluster_sizes[j]
            cluster_sizes.pop(i)
            cluster_sizes.pop(j)
            coverage = sum(cluster_size for cluster_size in cluster_sizes.values() if cluster_size >= 10) / self.data.shape[0]

            if coverage >= alpha:
                return distance


class FindConnectedComponents:
    def __init__(self, unlabeled_points, budget, adjacency_matrix, random_state=None):
        self.unlabeled_points = unlabeled_points
        self._max_budget = budget
        self.budget = budget
        self.adjacency_matrix = adjacency_matrix
        self._random_state = set_random_state(random_state)

        self.n_components, self.component_labels = self._find_connected_components()

        self.component_budgets = np.zeros((self._max_budget + 1, self.n_components)).astype(int)
        self._allot_component_budgets()
        self.component_budgets = {budget: self.component_budgets[budget] for budget in range(1, self._max_budget+1)}

    def _find_connected_components(self):
        """ :return n_components: number of connected components of the graph
            :return component_labels: corresponding connected component label of each data point """

        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(csgraph=self.adjacency_matrix, directed=False,
                                                              return_labels=True)

        return n_components, component_labels

    def __distribute_proportional_budget(self, budget, component_sizes, large_component_indices):
        # distribute component budget proportional to component size
        remaining_budget = budget - sum(self.component_budgets[budget])
        num_points = self.unlabeled_points.shape[0]
        for idx in range(self.n_components):
            self.component_budgets[budget][idx] += floor(component_sizes[idx] * remaining_budget / num_points)

        # if there is leftover budget, sample randomly proportional to size
        remaining_budget = int(budget - sum(self.component_budgets[budget]))  # recalculate remaining budget
        if remaining_budget > 0:
            total_points_in_large_components = sum(component_sizes[large_component_indices])
            distribution = component_sizes[large_component_indices] / total_points_in_large_components
            random_indices = [self._random_state.choice(large_component_indices, p=distribution) for _ in
                              range(remaining_budget)]
            self.component_budgets[budget][random_indices] += 1

    def _allot_component_budgets(self):
        _, component_sizes = np.unique(self.component_labels, return_counts=True)
        initialized_component_budgets = (component_sizes >= 10).astype(int)


        # if budget <= sum(initialized_component_budgets)
        ordered_component_sizes = sorted(zip(np.where(initialized_component_budgets == 1, component_sizes, 0),
                                             np.arange(self.n_components)), reverse=True)
        for budget in range(sum(initialized_component_budgets)+1):
            indices = [index for (_, index) in ordered_component_sizes][:budget]
            self.component_budgets[budget][indices] = 1

        # if budget > sum(initialized_component_budgets)
        large_component_indices = np.flatnonzero(initialized_component_budgets)  # indices of initialized_component_budgets that are nonzero
        for budget in range(sum(initialized_component_budgets)+1, self._max_budget+1):
            self.component_budgets[budget] = initialized_component_budgets
            self.__distribute_proportional_budget(budget, component_sizes, large_component_indices)


        return self.component_budgets


