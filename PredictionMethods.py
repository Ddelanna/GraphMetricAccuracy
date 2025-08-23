import sklearn
import numpy as np
import pandas as pd
from HelperFunctions import Plots

class EuclideanAccuracy: # 1nn algorithm using euclidean distance
    def __init__(self, sampling_model, oracle, create_plots=False):
        self.model = sampling_model
        self.oracle = oracle

        self.labeled_points = self.model.unlabeled_points.iloc[self.model.query_indices]
        self.queried_unlabeled_points = self.model.unlabeled_points.drop(index=self.model.query_indices)
        self.labels = pd.Series(self.oracle[self.labeled_points.index], index=self.model.query_indices)

        predicted_labels, confusion_matrix, self.score = self._calculate_score()

        if create_plots:
            title = f'{self.model}'
            Plots.plot_data(self.model.unlabeled_points, self.labeled_points, self.oracle,
                            self.labels, predicted_labels)
            Plots.plot_confusion_matrix(confusion_matrix, title)

    def __apply_KNN(self):
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(self.labeled_points, self.labels)
        predicted_labels = knn.predict(self.model.unlabeled_points)
        return predicted_labels

    def _calculate_score(self):
        predicted_labels = self.__apply_KNN()
        cm = sklearn.metrics.confusion_matrix(self.oracle, predicted_labels)
        accuracy = cm.diagonal().sum() / cm.sum()
        return predicted_labels, cm, accuracy


class GraphMetricAccuracy: # 1nn using graph metric
    def __init__(self, model, oracle, create_plots=False):
        self.model = model
        self.oracle = oracle

        self.labeled_points = self.model.unlabeled_points.iloc[self.model.query_indices]
        self.labels = pd.Series(self.oracle[self.model.query_indices], index=self.model.query_indices)
        self.queried_unlabeled_points = self.model.unlabeled_points.drop(index=self.model.query_indices)

        predicted_labels, confusion_matrix, self.score = self._calculate_score()

        if create_plots:
            title = f'{self.model}'
            self.plot_data(predicted_labels)
            Plots.plot_confusion_matrix(confusion_matrix, title)

    def __graph_predict(self):
        from scipy.sparse.csgraph import dijkstra
        from scipy.spatial.distance import squareform, pdist

        distance_matrix = squareform(pdist(self.model.unlabeled_points, metric='sqeuclidean'))
        distance_matrix[distance_matrix > 1.5] = np.inf
        self.weighted_adj_matrix = dijkstra(distance_matrix, indices=self.labeled_points.index, directed=False)

        # separate the points into connected and disconnected components
        self.inf_indices = np.isinf(self.weighted_adj_matrix).all(axis=0)
        non_inf_indices = np.invert(self.inf_indices)

        # propagate the labels on the connected points
        closest_labeled_pt_indices = np.argmin(self.weighted_adj_matrix, axis=0)
        predicted_labels = np.array([self.labels[self.labeled_points.index[idx]] for idx in closest_labeled_pt_indices])

        # label the disconnected points based on the connected points
        projection_indices = np.argmin(distance_matrix[np.ix_(non_inf_indices, self.inf_indices)], axis=0)
        predicted_labels[self.inf_indices] = predicted_labels[non_inf_indices][projection_indices]

        return predicted_labels

    def _calculate_score(self):
        from sklearn.metrics import confusion_matrix
        predicted_labels = self.__graph_predict()
        cm = confusion_matrix(self.oracle, predicted_labels)
        accuracy = cm.diagonal().sum() / cm.sum()
        return predicted_labels, cm, accuracy
