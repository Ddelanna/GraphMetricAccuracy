import sklearn
import numpy as np

import HelperFunctions
from HelperFunctions import Plots, AdjacencyMatrices


class EuclideanAccuracy:
    def __init__(self, unlabeled_points, query_indices, oracle, create_cm=False):
        self.unlabeled_points = unlabeled_points
        self.query_indices = query_indices
        self.oracle = oracle

        self.labeled_points = self.unlabeled_points.loc[self.query_indices]
        self.labels = self.oracle[self.labeled_points.index]
        self.queried_unlabeled_points = self.unlabeled_points.drop(index=self.query_indices)

        predicted_labels, confusion_matrix, self.score = self._calculate_score()

        if create_cm:
            Plots.plot_confusion_matrix(confusion_matrix)

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
    def __init__(self, unlabeled_points, query_indices, oracle, metric='2fermat', radius=None, create_cm=False):
        self.unlabeled_points = unlabeled_points
        self.query_indices = query_indices
        self.oracle = oracle
        self._metric = metric

        self._radius = radius
        if radius is None:
            radius = HelperFunctions.BestParameter(unlabeled_points).best_radius(0.975)
            self._radius = radius if '2' not in metric else radius**2

        self.labeled_points = self.unlabeled_points.loc[self.query_indices]
        self.labels = self.oracle[self.query_indices]
        self.queried_unlabeled_points = self.unlabeled_points.drop(index=self.query_indices)

        predicted_labels, confusion_matrix, self.score = self._calculate_score()

        if create_cm:
            Plots.plot_confusion_matrix(confusion_matrix)

    def __graph_predict(self):
        from scipy.sparse.csgraph import dijkstra

        distance_matrix = AdjacencyMatrices().distance_matrix(self.unlabeled_points, metric=self._metric)
        distance_matrix[distance_matrix > self._radius] = np.inf # DO NOT REMOVE
        indices = [self.unlabeled_points.index.get_loc(idx) for idx in self.query_indices] # get iloc from loc
        self.weighted_adj_matrix = dijkstra(distance_matrix, indices=indices, directed=False)

        # separate the points into connected and disconnected components
        inf_indices = np.isinf(self.weighted_adj_matrix).all(axis=0)
        non_inf_indices = np.invert(inf_indices)

        # propagate the labels on the connected points
        closest_labeled_pt_indices = np.argmin(self.weighted_adj_matrix, axis=0)
        predicted_labels = np.array(self.labels.iloc[closest_labeled_pt_indices])

        # label the disconnected points based on the closest connected points
        projection_indices = np.ravel(np.argmin(distance_matrix[np.ix_(non_inf_indices, inf_indices)], axis=0))
        # note to self: np.ravel flattens projection_indices from 2 dimensions to 1

        predicted_labels[inf_indices] = predicted_labels[non_inf_indices][projection_indices]

        return predicted_labels

    def _calculate_score(self):
        from sklearn.metrics import confusion_matrix
        predicted_labels = self.__graph_predict()
        cm = confusion_matrix(self.oracle, predicted_labels)
        accuracy = cm.diagonal().sum() / cm.sum()
        return predicted_labels, cm, accuracy


class NearestNeighborAccuracy:
    def __init__(self, unlabeled_points, query_indices, oracle, radius=None, metric=None, create_plots=False):
        if metric == 'euclidean':
            self.score = EuclideanAccuracy(unlabeled_points, query_indices, oracle, create_cm=create_plots).score
        elif 'fermat' in metric:
            self.score = GraphMetricAccuracy(unlabeled_points, query_indices, oracle, metric, radius, create_cm=create_plots).score



