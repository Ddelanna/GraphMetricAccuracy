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
    def _check_sparsity(matrix, sparse):
        if not sparse:
            return matrix
        from scipy.sparse import csc_matrix
        return csc_matrix(matrix)

    @staticmethod
    def distance_matrix(data, metric='euclidean'):
        from scipy.spatial.distance import squareform, pdist

        if (metric == 'euclidean') or (('fermat' in metric) and ('1' in metric)):
            return squareform(pdist(data, metric='euclidean'))
        elif metric == 'sqeuclidean':
            return squareform(pdist(data, metric='sqeuclidean'))
        elif ('fermat' in metric) and ('2' in metric):
            distance_matrix = squareform(pdist(data, metric='sqeuclidean'))
            from scipy.sparse.csgraph import dijkstra
            return dijkstra(distance_matrix, directed=False)
        else:
            raise ValueError('Metric must be either \'euclidean\', \'sqeuclidean\', or \'pfermat\'.')

    def knn_graph(self, data, k=1, metric='euclidean', sparse=True):
        from sklearn.neighbors import kneighbors_graph
        knn_graph = kneighbors_graph(data, n_neighbors=k, metric=metric, mode='connectivity').toarray()
        knn_graph = np.maximum(knn_graph, knn_graph.transpose())
        return self._check_sparsity(knn_graph, sparse)

    def binary_epsilon_graph(self, data, radius=1.0, metric='euclidean', sparse=True):
        distance_matrix = self.distance_matrix(data, metric=metric)
        binary_epsilon_graph = (distance_matrix <= radius).astype(int)
        return self._check_sparsity(binary_epsilon_graph, sparse)


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








