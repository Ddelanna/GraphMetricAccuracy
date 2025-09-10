import time
from CreateData import *
import PredictionMethods
from SamplingAlgorithms import *
from HelperFunctions import AdjacencyMatrices as AM
from HelperFunctions import BestParameter


class Iterate:
    def __init__(self):
        self.data_sets = [('spiral', create_spiral_data)]

        self.models = [(RandomSampling, None),
                       (KmeansSampling, 'full'),
                       (ProbCoverSampling, 'epsilon'),
                       (ProbCoverSampling, 'knn'),
                       (ConnectedComponentSampling, 'epsilon'),
                       (ConnectedComponentSampling, 'knn')]

    def _build_graph(self, data, graph_method, metric, radius=None):
        if graph_method == 'full':
            return AM().full_graph(data, metric=metric)
        elif graph_method == 'epsilon':
            return AM().binary_epsilon_graph(data, radius=radius, metric=metric)
        elif graph_method == 'knn':
            return AM().knn_graph(data, k=1, metric=metric)
        else:
            return None

    def main(self): # todo: parallelize
        score_df = pd.DataFrame(columns=['Budget', 'Score', 'Metric', 'Graph Type', 'Sampling Model', 'Prediction Method', 'Radius', 'Computation Time'])

        for name, data_set in self.data_sets:
            for budget in [10, 20, 30]:
                for metric in ['2fermat', '1fermat', 'euclidean']:
                    for model, graph_method in self.models:
                        graph_metric_scores, euclidean_scores = [], []
                        for seed in range(1, 3):
                            data, oracle = data_set(100, random_state=seed)
                            if seed == 1:
                                time1 = time.time()
                                radius = BestParameter(data, budget, metric).best_radius(alpha=0.95)
                                radius_time = time.time() - time1

                            time2 = time.time()
                            graph = self._build_graph(data, graph_method, metric, radius=radius)
                            query_indices = model(data, budget, graph, random_state=1).query_indices

                            p=1
                            if metric == '2fermat':
                                p = 2

                            time3 = time.time()
                            graph_metric_scores.append(PredictionMethods.GraphMetricAccuracy(data, query_indices, oracle, fermat_p=p, radius=radius).score)
                            graph_time = time.time() - time2
                            time4 = time.time()
                            euclidean_scores.append(PredictionMethods.EuclideanAccuracy(data, query_indices, oracle).score)
                            euclidean_time = time.time() - time4 + (time3 - time2)

                        new_data_point = pd.DataFrame([{'Budget': budget,
                                                        'Score': np.round(np.average(graph_metric_scores), 3),
                                                        'Metric': metric,
                                                        'Graph Type': graph_method,
                                                        'Sampling Model': model,
                                                        'Prediction Method': 'Graph Metric',
                                                        'Radius': radius,
                                                        'Computation Time': graph_time + radius_time,
                                                        }])
                        score_df = pd.concat([score_df, new_data_point])

                        new_data_point = pd.DataFrame([{'Budget': budget,
                                                        'Score': np.round(np.average(euclidean_scores), 3),
                                                        'Metric': metric,
                                                        'Graph Type': graph_method,
                                                        'Sampling Model': model,
                                                        'Prediction Method': 'Euclidean',
                                                        'Radius': radius,
                                                        'Computation Time': euclidean_time + radius_time,
                                                        }])
                        print(new_data_point)
                        score_df = pd.concat([score_df, new_data_point])

            score_df.to_csv(f'[{name}]_scores.csv', index=False)
