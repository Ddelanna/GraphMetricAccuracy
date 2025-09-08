import sklearn
import numpy as np
import pandas as pd
from HelperFunctions import Plots, AdjacencyMatrices

class EuclideanAccuracy:
    def __init__(self, unlabeled_points, query_indices, oracle, radius=None, create_plots=False):
        self.unlabeled_points = unlabeled_points
        self.query_indices = query_indices
        self.oracle = oracle

        self.labeled_points = self.unlabeled_points.loc[self.query_indices]
        self.queried_unlabeled_points = self.unlabeled_points.drop(index=self.query_indices)
        self.labels = pd.Series(self.oracle[self.labeled_points.index], index=self.query_indices)

        predicted_labels, confusion_matrix, self.score = self._calculate_score()

        if create_plots:
            Plots.plot_data(self.unlabeled_points, self.labeled_points, self.oracle,
                            self.labels, predicted_labels)
            Plots.plot_confusion_matrix(confusion_matrix, title=None)

    def __apply_KNN(self):
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(self.labeled_points, self.labels)
        predicted_labels = knn.predict(self.unlabeled_points)
        return predicted_labels

    def _calculate_score(self):
        predicted_labels = self.__apply_KNN()
        cm = sklearn.metrics.confusion_matrix(self.oracle, predicted_labels)
        accuracy = cm.diagonal().sum() / cm.sum()
        return predicted_labels, cm, accuracy


class GraphMetricAccuracy:
    def __init__(self, unlabeled_points, query_indices, oracle, radius=1.0, create_plots=False):
        self.unlabeled_points = unlabeled_points
        self.query_indices = query_indices
        self.oracle = oracle
        self._radius = radius

        self.labeled_points = self.unlabeled_points.loc[self.query_indices]
        self.labels = pd.Series(self.oracle[self.query_indices], index=self.query_indices)
        self.queried_unlabeled_points = self.unlabeled_points.drop(index=self.query_indices)

        predicted_labels, confusion_matrix, self.score = self._calculate_score()

        if create_plots:
            Plots.plot_data(self.unlabeled_points, self.labeled_points, self.oracle,
                            self.labels, predicted_labels)
            Plots.plot_confusion_matrix(confusion_matrix, title=None)

    def __graph_predict(self):
        from scipy.sparse.csgraph import dijkstra

        distance_matrix = AdjacencyMatrices().distance_matrix(self.unlabeled_points, metric='sqeuclidean')
        distance_matrix[distance_matrix > self._radius] = np.inf
        indices = [self.unlabeled_points.index.get_loc(label) for label in list(self.labeled_points.index)]
        self.weighted_adj_matrix = dijkstra(distance_matrix, indices=indices, directed=False)

        # separate the points into connected and disconnected components
        self.inf_indices = np.isinf(self.weighted_adj_matrix).all(axis=0)
        non_inf_indices = np.invert(self.inf_indices)

        # propagate the labels on the connected points
        closest_labeled_pt_indices = np.argmin(self.weighted_adj_matrix, axis=0)
        predicted_labels = np.array([self.labels[self.labeled_points.index[idx]] for idx in closest_labeled_pt_indices])

        # label the disconnected points based on the closest connected points
        projection_indices = np.ravel(np.argmin(distance_matrix[np.ix_(non_inf_indices, self.inf_indices)], axis=0))
        # np.ravel flattens projection_indices from 2 dimensions to 1

        predicted_labels[self.inf_indices] = predicted_labels[non_inf_indices][projection_indices]

        return predicted_labels

    def _calculate_score(self):
        from sklearn.metrics import confusion_matrix
        predicted_labels = self.__graph_predict()
        cm = confusion_matrix(self.oracle, predicted_labels)
        accuracy = cm.diagonal().sum() / cm.sum()
        return predicted_labels, cm, accuracy
