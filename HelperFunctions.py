import matplotlib.pyplot as plt
import numpy as np
import sklearn


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

# class BestParameter:
#     def __init__(self, data, budget, metric='euclidean'):
#         self.data = data
#         self.budget = budget
#
#         self.distance_matrix = AdjacencyMatrices().distance_matrix(data, metric=metric) # todo
#
#     def _compute_coverage(self, radius):
#         """ :return coverage: the ratio of points in the same connected component as a labeled point
#             to the total number of points """
#
#         adjacency_matrix = (self.distance_matrix <= radius).astype(int)
#         from scipy.sparse.csgraph import connected_components
#         n_components, component_labels = connected_components(csgraph=adjacency_matrix, directed=False, return_labels=True)
#         _, component_sizes = np.unique(component_labels, return_counts=True)
#         ordered_components_sizes = sorted(component_sizes, reverse=True)
#         coverage = sum(ordered_components_sizes[:self.budget]) / self.data.shape[0]
#
#         return coverage
#
#     def best_radius(self, alpha):
#         diameter = np.max(self.distance_matrix)
#         tolerance = 0.01  # todo : how to determine tolerance?
#         radius_lowerbound, radius_upperbound = 0, diameter
#
#         while (radius_upperbound - radius_lowerbound) > tolerance:
#
#             radius_to_compute = (radius_lowerbound + radius_upperbound) / 2
#
#             coverage = self._compute_coverage(radius_to_compute)
#             if coverage > alpha:
#                 radius_upperbound = radius_to_compute
#             else:
#                 radius_lowerbound = radius_to_compute
#
#         best_radius = (radius_lowerbound + radius_upperbound) / 2
#         return best_radius

class BestParameter:
    def __init__(self, data, budget):
        self.data = data
        self.budget = budget

        Sum_of_squared_distances = []
        K = range(1, 15)
        for k in K:
            km = sklearn.cluster.KMeans(n_clusters=k)
            km = km.fit(self.data)
            Sum_of_squared_distances.append(km.inertia_)

        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

        self.optimal_n_components = 5

        self.distance_matrix = AdjacencyMatrices().distance_matrix(data, metric='euclidean')

    def _compute_coverage(self, radius):
        """ :return coverage: the ratio of points in the same connected component as a labeled point
            to the total number of points """

        adjacency_matrix = (self.distance_matrix <= radius).astype(int)
        from scipy.sparse.csgraph import connected_components
        n_components = connected_components(csgraph=adjacency_matrix, directed=False, return_labels=False)

        # _, component_sizes = np.unique(component_labels, return_counts=True)
        # ordered_component_sizes = sorted(component_sizes, reverse=True)
        # coverage = sum(ordered_component_sizes[:self.budget]) / self.data.shape[0]

        return n_components

    def best_radius(self, alpha):
        diameter = np.max(self.distance_matrix)
        tolerance = 0.01  # todo : how to determine tolerance?
        radius_lowerbound, radius_upperbound = 0, diameter

        while (radius_upperbound - radius_lowerbound) > tolerance:

            radius_to_compute = (radius_lowerbound + radius_upperbound) / 2

            n_components = self._compute_coverage(radius_to_compute)
            if n_components > self.optimal_n_components:
                radius_lowerbound = radius_to_compute
            else:
                radius_upperbound = radius_to_compute

        best_radius = (radius_lowerbound + radius_upperbound) / 2
        return best_radius


class FindConnectedComponents:
    def __init__(self, unlabeled_points, budget, adjacency_matrix, random_state=None):
        self.unlabeled_points = unlabeled_points
        self.budget = budget
        self.adjacency_matrix = adjacency_matrix
        self._random_state = set_random_state(random_state)

        self.n_components, self.component_labels = self._find_connected_components()
        self.component_budgets = self._allot_component_budgets()

    def _find_connected_components(self):
        """ :return n_components: number of connected components of the graph
            :return component_labels: corresponding connected component label of each data point """

        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(csgraph=self.adjacency_matrix, directed=False,
                                                              return_labels=True)

        return n_components, component_labels

    def __sample_from_largest_components(self, component_sizes):
        """ While we are under budget, sample one point from the k largest connected components
            until the connected components are too small """

        component_budgets = [0 for _ in range(self.n_components)]
        large_component_index = []

        ordered_components_by_size = sorted(zip(component_sizes, np.arange(self.n_components)), reverse=True)
        for component_size, component_idx in ordered_components_by_size:
            # keep going until the (clusters are too small) or until (budget is used up)
            if (component_size <= 10) or (sum(component_budgets) == self.budget):
                break
            component_budgets[component_idx] += 1
            large_component_index.append(component_idx)

        return component_budgets, large_component_index

    def _allot_component_budgets(self):
        """ :return component_budgets: the budget allotted for each component in the order of component labels """

        num_points = self.unlabeled_points.shape[0]
        _, component_sizes = np.unique(self.component_labels, return_counts=True)

        # ensure all large components have budget >= 1
        component_budgets, large_component_index = self.__sample_from_largest_components(component_sizes)

        # allot budget proportionally to component size
        from math import floor
        remaining_budget = self.budget - sum(component_budgets)
        for idx in range(self.n_components):
            component_budgets[idx] += floor(component_sizes[idx] / num_points * remaining_budget)

        # if there is leftover budget, sample randomly proportional to size
        total_points_in_large_components = sum([component_sizes[idx] for idx in large_component_index])
        if sum(component_budgets) < self.budget:
            distribution = [component_sizes[idx] / total_points_in_large_components for idx in large_component_index]
            for _ in range(self.budget - sum(component_budgets)):
                component_budgets[self._random_state.choice(large_component_index, p=distribution)] += 1

        return component_budgets











