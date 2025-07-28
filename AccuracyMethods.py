import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd



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


class KnnAccuracy:
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


class GraphMetricAccuracy:
    def __init__(self, model, oracle, create_plots=False):
        self.model = model
        self.oracle = oracle

        self.labeled_points = self.model.unlabeled_points.iloc[self.model.query_indices]
        self.queried_unlabeled_points = self.model.unlabeled_points.drop(index=self.model.query_indices)
        self.labels = pd.Series(self.oracle[self.labeled_points.index], index=self.model.query_indices)

        predicted_labels, confusion_matrix, self.score = self._calculate_score()

        if create_plots:
            title = f'{self.model}'
            self.plot_data(predicted_labels)
            Plots.plot_confusion_matrix(confusion_matrix, title)

    def __graph_predict(self):
        from scipy.sparse.csgraph import dijkstra
        squared_dist_matrix = self.model.distance_matrix ** 2
        squared_dist_matrix[squared_dist_matrix > self.model.radius] = np.inf
        self.weighted_adj_matrix = dijkstra(squared_dist_matrix, indices=self.labeled_points.index, directed=False)

        # separate the points into connected and disconnected components
        self.inf_indices = np.isinf(self.weighted_adj_matrix).all(axis=0)
        non_inf_indices = np.invert(self.inf_indices)

        # propagate the labels on the connected points
        closest_labeled_pt_indices = np.argmin(self.weighted_adj_matrix, axis=0)
        predicted_labels = np.array([self.labels[self.labeled_points.index[idx]] for idx in closest_labeled_pt_indices])

        # label the disconnected points based on the connected points
        projection_indices = np.argmin(self.model.distance_matrix[np.ix_(non_inf_indices, self.inf_indices)], axis=0)
        predicted_labels[self.inf_indices] = predicted_labels[non_inf_indices][projection_indices]

        return predicted_labels

    def _calculate_score(self):
        import sklearn
        predicted_labels = self.__graph_predict()
        cm = sklearn.metrics.confusion_matrix(self.oracle, predicted_labels)
        accuracy = cm.diagonal().sum() / cm.sum()
        return predicted_labels, cm, accuracy

    def plot_data(self, predicted_labels):
        """ Creates a 2D plot based on first two columns of data. """
        from matplotlib import pyplot as plt
        import matplotlib

        colors = ["blue", "orange", "green", "purple", "black"]

        unlabeled_points = np.array(self.model.unlabeled_points.iloc[:,[0, 1]])
        labeled_points = np.array(self.labeled_points.iloc[:,[0, 1]])

        fig, ax = plt.subplots()

        for i in range(len(colors)):
            for (x, y) in zip(labeled_points[self.labels == i, 0], labeled_points[self.labels == i, 1]):
                circ = matplotlib.patches.Circle((x, y), radius=self.model.radius, color=colors[i], alpha=0.25)
                ax.add_patch(circ)

            ax.scatter(unlabeled_points[self.oracle == i, 0], unlabeled_points[self.oracle == i, 1],
                       color=colors[i], label=f"Cluster{i}", s=1.0)

        for i in range(len(predicted_labels)):
            if predicted_labels[i] != self.oracle[i]:
                ax.scatter(unlabeled_points[i, 0], unlabeled_points[i, 1],
                           color='red', s=30.0, alpha=0.5)

        ax.scatter(unlabeled_points[self.inf_indices, 0], unlabeled_points[self.inf_indices, 1],
                    color='purple', s=30.0, alpha=0.5)

        fig.show()
