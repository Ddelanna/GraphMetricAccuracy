import matplotlib.pyplot as plt
import numpy as np
import sklearn
import graphlearning as gl
import time


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
    def plot_scores(x, y_avg, y_std=None, title=''):
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
        distance_matrix = squareform(pdist(data, metric=metric))
        return distance_matrix

    def knn_graph(self, data, k=1, metric='euclidean'):
        return gl.weightmatrix.knn(data, k=k, kernel='distance', similarity=metric)

    def binary_epsilon_graph(self, data, radius=1.0, metric='euclidean'):
        distance_matrix = self.distance_matrix(data, metric=metric)
        return (distance_matrix <= radius).astype(int)

    def weighted_epsilon_graph(self, data, radius=1.0, metric='euclidean'):
        if metric == 'graph_euclidean' or metric == 'graph_sqeuclidean':
            from scipy.sparse.csgraph import dijkstra
            distance_matrix = self.distance_matrix(data, metric=metric[6:])
            distance_matrix[distance_matrix > radius] = np.inf
            weighted_adj_matrix = dijkstra(distance_matrix, directed=False)
        else:
            weighted_adj_matrix = gl.weightmatrix.epsilon_ball(data, epsilon=radius, kernel='distance')

        return weighted_adj_matrix

    # todo : combine with distance_matrix()?
    def compute_distance(self, W, boundary_set, distance_type='peikonal'):
        if distance_type == 'peikonal':
            return gl.graph(W).peikonal(bdy_set=boundary_set, p=1)
        elif distance_type == 'euclidean':
            return W
        elif 'fermat' in distance_type:  # distance_type should be in the format 'fermatp' (e.g. 'fermat2')
            p = int(distance_type[6:]) # drop 'fermat'
            return gl.graph(W**p).dijkstra(bdy_set=boundary_set, bdy_val=0)
        else:
            raise ValueError(f"Unsupported distance method: {distance_type}")






