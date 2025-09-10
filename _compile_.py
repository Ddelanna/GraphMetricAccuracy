import time
from CreateData import *
from PredictionMethods import EuclideanAccuracy, GraphMetricAccuracy
from SamplingAlgorithms import *
from HelperFunctions import AdjacencyMatrices as AM
from HelperFunctions import BestParameter
from joblib import Parallel, delayed

class Iterate:
    def __init__(self):
        self._num_points = 200  # num_points = None gets the entire data set
        self._num_iters = 2 # number of runs to average the score over
        self._data_generators = [(create_spiral_data, 'spiral')] # (generator, file_name)
        self._budgets = [10, 20, 30]
        
        self._search_grid = self.build_search_grid()

        self.create_score_csv()

    def build_search_grid(self):
        import itertools

        query_models = [(RandomSampling, None),
                  (KmeansSampling, 'full'),
                  (ProbCoverSampling, 'epsilon'),
                  (ProbCoverSampling, 'knn'),
                  (ConnectedComponentSampling, 'epsilon'),
                  (ConnectedComponentSampling, 'knn')]
        metric = ['2fermat', '1fermat', 'euclidean']
        prediction_models = [GraphMetricAccuracy, EuclideanAccuracy]

        return list(itertools.product(query_models, self._budgets, metric, prediction_models))

    @staticmethod
    def __build_graph(data, graph_method, metric, radius=None):
        if graph_method == 'full':
            return AM().full_graph(data, metric=metric)
        elif graph_method == 'epsilon':
            return AM().binary_epsilon_graph(data, radius=radius, metric=metric)
        elif graph_method == 'knn':
            return AM().knn_graph(data, k=10, metric=metric)
        else:
            return None

    def __compute_score(self, data_generator, budget, metric, graph_method, query_model, prediction_model):
        scores, computation_times = [], []
        for seed in range(1, self._num_iters + 1):
            data, oracle = data_generator[0](self._num_points, random_state=seed)

            start_time = time.time()

            radius = BestParameter(data, budget, metric).best_radius(alpha=0.95) # todo: radius not always necessary
            graph = self.__build_graph(data, graph_method, metric, radius=radius)
            query_indices = query_model(data, budget, graph, random_state=seed).query_indices
            scores.append(prediction_model(data, query_indices, oracle, radius=radius, metric=metric).score * 100)

            computation_times.append(time.time() - start_time)

        return np.round(np.average(scores), 3), np.round(np.average(computation_times), 3)

    def _get_new_data_point(self, data_generator, budget, metric, graph_method, query_model, prediction_model):
        average_score, average_computation_time = self.__compute_score(data_generator, budget, metric, graph_method,
                                                                       query_model, prediction_model)

        new_data_point = pd.DataFrame([{'Budget': budget,
                                        'Score': average_score,
                                        'Metric': metric,
                                        'Graph Type': graph_method,
                                        'Sampling Model': str(query_model)[27:-2],
                                        'Prediction Method': str(prediction_model)[26:-2],
                                        'Computation Time': average_computation_time,
                                        }])

        return new_data_point

    def create_score_csv(self):
        start_time = time.time()
        for data_generator in self._data_generators:
            score_df = pd.DataFrame(columns=['Budget', 'Score', 'Metric', 'Graph Type', 'Sampling Model',
                                              'Prediction Method', 'Computation Time'])

            scores = Parallel(n_jobs=-1)(
                delayed(self._get_new_data_point)(query_model, graph_method, data_generator, budget, metric, prediction_model)
                for (query_model, graph_method), budget, metric, prediction_model in self._search_grid
            )
            for data_point in scores:
                score_df = pd.concat([score_df, data_point])


            score_df.to_csv(f'results/[{data_generator[1]}]_scores.csv', index=False)
        print('Total Run Time', time.time() - start_time)
