import time
from CreateData import *
from PredictionModels import EuclideanAccuracy, GraphMetricAccuracy, NearestNeighborAccuracy
from QueryModels import *
from HelperFunctions import AdjacencyMatrices as AM
from HelperFunctions import BestParameter
from joblib import Parallel, delayed

class Pipeline:
    def __init__(self):
        self._num_points = 100  # num_points = None gets the entire data set
        self._num_iters = 2 # number of runs to average the score over
        self._data_generators = [(create_spiral_data, 'spiral')] # (generator, file_name)
        self._budgets = [10, 20]

        self._search_grid = self.build_search_grid()

        self.create_score_csv()

    def build_search_grid(self):
        import itertools

        query_model_parameters = [(RandomSampling, None, None),
                        (KmeansSampling, 'full', 'euclidean'),
                        (KmeansSampling, 'full', '2fermat'),
                        (ProbCoverSampling, 'epsilon', 'euclidean'),
                        (ProbCoverSampling, 'epsilon', '1fermat'),
                        (ProbCoverSampling, 'epsilon', '2fermat'),
                        (ProbCoverSampling, 'knn', 'euclidean'),
                        (ProbCoverSampling, 'knn', 'sqeuclidean'),
                        (ConnectedComponentSampling, 'epsilon', 'euclidean'),
                        (ConnectedComponentSampling, 'epsilon', '1fermat'),
                        (ConnectedComponentSampling, 'epsilon', '2fermat'),
                        (ConnectedComponentSampling, 'knn', 'euclidean'),
                        (ConnectedComponentSampling, 'knn', 'sqeuclidean')]
        prediction_models = ['euclidean', '1fermat', '2fermat']

        return list(itertools.product(self._data_generators, self._budgets, query_model_parameters, prediction_models))

    @staticmethod
    def __build_graph(data, budget, graph_method, metric):
        if graph_method == 'full':
            return AM().distance_matrix(data, metric=metric)
        elif graph_method == 'epsilon':
            radius = BestParameter(data, budget, metric).best_radius(alpha=0.95)
            return AM().binary_epsilon_graph(data, radius=radius, metric=metric)
        elif graph_method == 'knn':
            return AM().knn_graph(data, k=15, metric=metric)
        else:
            return None

    def __compute_score(self, data_generator, budget, query_model, graph_method, metric, prediction_model):
        scores, computation_times = [], []
        for seed in range(1, self._num_iters + 1):
            data, oracle = data_generator[0](self._num_points, random_state=seed)

            start_time = time.time()

            graph = self.__build_graph(data, budget, graph_method, metric)
            query_indices = query_model(data, budget, graph, random_state=seed).query_indices
            score = NearestNeighborAccuracy(data, query_indices, oracle, metric=prediction_model).score
            scores.append(score * 100)

            computation_times.append(time.time() - start_time)

        return np.round(np.average(scores), 3), np.round(np.average(computation_times), 3)

    def _get_new_data_point(self, data_generator, budget, query_model, graph_method, metric, prediction_model):
        average_score, average_computation_time = self.__compute_score(data_generator, budget, query_model,
                                                                       graph_method, metric, prediction_model)

        new_data_point = pd.DataFrame([{'Budget': budget,
                                        'Score': average_score,
                                        'Graph Type': f'{graph_method}-{metric}',
                                        'Sampling Model': str(query_model),
                                        'Prediction Method': prediction_model,
                                        'Computation Time': average_computation_time,
                                        }])

        return new_data_point

    def create_score_csv(self):
        start_time = time.time()
        for data_generator in self._data_generators:
            score_df = pd.DataFrame(columns=['Budget', 'Score', 'Graph Type', 'Sampling Model',
                                              'Prediction Method', 'Computation Time'])

            scores = Parallel(n_jobs=-1)(
                delayed(self._get_new_data_point)(data_generator, budget, query_model, graph_method, metric, prediction_model)
                for data_generator, budget, (query_model, graph_method, metric), prediction_model in self._search_grid
            )
            for data_point in scores:
                score_df = pd.concat([score_df, data_point])


            score_df.to_csv(f'results/[{data_generator[1]}]_scores.csv', index=False)
        print('Total Run Time', time.time() - start_time)