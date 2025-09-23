import matplotlib.pyplot as plt
import numpy as np
import sklearn
import graphlearning as gl


def _set_random_state(random_state):
    if isinstance(random_state, int):
        return np.random.default_rng(random_state)
    return np.random.default_rng(np.random.random_integers(low=0, high=10000))


class Plots:
    @staticmethod
    def plot_data(data_points, labeled_points, oracle, labels, predicted_labels):
        colors = ["blue", "orange", "green", "purple", "black"]

        data_points = np.array(data_points)
        labeled_points = np.array(labeled_points)

        for i in range(len(colors)):
            plt.scatter(labeled_points[labels == i, 0], labeled_points[labels == i, 1],
                        color=colors[i], s=30.0, alpha=0.5)

        for i in range(len(colors)):
            plt.scatter(data_points[oracle == i, 0], data_points[oracle == i, 1],
                        color=colors[i], label=f"Cluster{i}", s=1.0)

        for i in range(len(predicted_labels)):
            if predicted_labels[i] != oracle[i]:
                plt.scatter(data_points[i, 0], data_points[i, 1],
                            color='red', s=30.0, alpha=0.5)

        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, title):
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

    @staticmethod
    def plot_connected_components(data_by_component, component_sizes):
        from matplotlib import pyplot as plt
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, 10))
        fig, ax = plt.subplots()
        color_idx = 0

        for component_label in range(len(component_sizes)):
            if component_sizes[component_label] > 10:
                ax.scatter(np.array(data_by_component[component_label])[:, 0],
                           np.array(data_by_component[component_label])[:, 1],
                           color=colors[color_idx], s=1.0)
                color_idx += 1
            else:
                ax.scatter(np.array(data_by_component[component_label])[:, 0],
                           np.array(data_by_component[component_label])[:, 1],
                           color='black', s=1.0)

        fig.show()


class AdjacencyMatrices:
    @staticmethod
    def distance_matrix(data, metric='euclidean'):
        from scipy.spatial.distance import squareform, pdist

        if 'fermat' in metric: # should be in format 'pfermat' where p is power
            p = int(metric[0])
            distance_matrix = squareform(pdist(data, metric='euclidean')) ** p
            from scipy.sparse.csgraph import dijkstra
            distance_matrix = dijkstra(distance_matrix, directed=False)
        else:
            distance_matrix = squareform(pdist(data, metric=metric))

        return distance_matrix

    @staticmethod
    def knn_graph(data, k=1, metric='euclidean', sparse=True):
        knn_graph = gl.weightmatrix.knn(data.to_numpy(), k=k, kernel='distance', similarity='euclidean').toarray()
        knn_graph[knn_graph > 0.0] = 1

        # from sklearn.neighbors import kneighbors_graph
        # knn_graph = kneighbors_graph(data, n_neighbors=k, metric=metric, mode='connectivity')
        # print(knn_graph)

        if sparse:
            from scipy.sparse import csc_matrix
            return csc_matrix(knn_graph)

        return knn_graph

    def binary_epsilon_graph(self, data, radius=1.0, metric='euclidean', sparse=True):
        distance_matrix = self.distance_matrix(data, metric=metric)
        binary_epsilon_graph = (distance_matrix <= radius).astype(int)
        if sparse:
            from scipy.sparse import csc_matrix
            return csc_matrix(binary_epsilon_graph)
        return binary_epsilon_graph

    def weighted_epsilon_graph(self, data, radius=1.0, metric='euclidean'):
        if metric == 'graph_euclidean' or metric == 'graph_sqeuclidean':
            from scipy.sparse.csgraph import dijkstra
            distance_matrix = self.distance_matrix(data, metric=metric[6:])
            distance_matrix[distance_matrix > radius] = np.inf
            weighted_adj_matrix = dijkstra(distance_matrix, directed=False)
        else:
            weighted_adj_matrix = gl.weightmatrix.epsilon_ball(data, epsilon=radius, kernel='distance')

        return weighted_adj_matrix

    def full_graph(self, data, metric='euclidean'):
        distance_matrix = self.distance_matrix(data, metric=metric)
        return gl.graph(distance_matrix)


class BestParameter:
    def __init__(self, data, budget, metric='euclidean'):
        self.data = data
        self.budget = budget

        self.distance_matrix = AdjacencyMatrices().distance_matrix(data, metric=metric) # todo

    def _compute_coverage(self, radius):
        """ :return coverage: the ratio of points in the same connected component as a labeled point
            to the total number of points """

        adjacency_matrix = (self.distance_matrix <= radius).astype(int)
        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(csgraph=adjacency_matrix, directed=False, return_labels=True)
        _, component_sizes = np.unique(component_labels, return_counts=True)
        ordered_components_sizes = sorted(component_sizes, reverse=True)
        coverage = sum(ordered_components_sizes[:self.budget]) / self.data.shape[0]

        return coverage

    def best_radius(self, alpha):
        diameter = np.max(self.distance_matrix)
        tolerance = 0.01  # todo : how to determine tolerance?
        radius_lowerbound, radius_upperbound = 0, diameter

        while (radius_upperbound - radius_lowerbound) > tolerance:

            radius_to_compute = (radius_lowerbound + radius_upperbound) / 2

            coverage = self._compute_coverage(radius_to_compute)
            if coverage > alpha:
                radius_upperbound = radius_to_compute
            else:
                radius_lowerbound = radius_to_compute

        best_radius = (radius_lowerbound + radius_upperbound) / 2
        return best_radius








