import time
from PredictionModels import NearestNeighborAccuracy
from QueryModels import *
from HelperFunctions import AdjacencyMatrices
from HelperFunctions import BestParameter, FindConnectedComponents
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import chain



class SummaryResults:
    def __init__(self, data_generators):
        self._num_points = 1000  # num_points = None gets the entire data set
        self._num_iters = 2 # number of runs to average the score over
        self._data_generators = data_generators
        self._budgets = [10, 20, 30, 40]

        self.create_score_csv()

    def _build_search_grid(self):
        import itertools

        seeds = np.arange(self._num_iters)
        find_connected_components = ['epsilon', 'knn', False]
        query_model_parameters = [(RandomSampling, None, None),
                        (KmeansSampling, None, 'euclidean'),
                        (GraphKmeansSampling, 'full', '1fermat'),
                        (GraphKmeansSampling, 'full', '2fermat'),
                        (ProbCoverSampling, 'epsilon', 'euclidean'),
                        (ProbCoverSampling, 'epsilon', '1fermat'),
                        (ProbCoverSampling, 'epsilon', '2fermat')]
        prediction_models = ['euclidean', '1fermat', '2fermat']

        search_grid = itertools.product(seeds, self._data_generators, find_connected_components,
                                        query_model_parameters, prediction_models)
        return list(search_grid)

    @staticmethod
    def __build_graph(data, graph_method, metric, alpha=1.0):
        if graph_method == 'full':
            return AdjacencyMatrices().distance_matrix(data, metric=metric)
        elif graph_method == 'epsilon':
            radius = BestParameter(data).best_radius(alpha=alpha)
            radius = radius if '2' not in metric else radius**2
            return AdjacencyMatrices().epsilon_graph(data, radius=radius, metric=metric)
        elif graph_method == 'knn':
            k = 5 # todo: find best k
            return AdjacencyMatrices().knn_graph(data, k=k, metric=metric)
        else:
            return None

    def __get_query_indices(self, search_parameters):
        new_data_points = []
        seed, data_generator, find_connected_component, (query_model, graph_method, metric), prediction_model = search_parameters

        data, oracle = data_generator[0](self._num_points, random_state=seed)
        graph = self.__build_graph(data, graph_method, metric, alpha=0.975)
        query_indices = query_model(data, max(self._budgets), graph, random_state=seed).query_indices

        for budget in self._budgets:
            score = NearestNeighborAccuracy(data, query_indices[:budget], oracle, metric=prediction_model).score

            new_data_points.append([np.round(100 * score), time.time(), budget, find_connected_component,
                                    f'{graph_method}-{metric}', str(query_model), prediction_model])

        return new_data_points

    def __apply_query_model_to_component(self, seed, data, budget, sampling_model, CC, component_label):
        query_model, graph_method, metric = sampling_model
        component_data = data[CC.component_labels == component_label]
        component_budget = CC.component_budgets[budget][component_label]
        component_graph = self.__build_graph(component_data, graph_method, metric, alpha=0.975)
        query_indices = query_model(component_data, component_budget, component_graph, random_state=seed).query_indices
        return query_indices

    def __get_query_indices_using_connected_components(self, search_parameters):
        new_data_points = []
        seed, data_generator, find_connected_component, sampling_model, prediction_model = search_parameters

        data, oracle = data_generator[0](self._num_points, random_state=seed)
        outer_graph = self.__build_graph(data, find_connected_component, 'euclidean', alpha=0.90)
        CC = FindConnectedComponents(data, max(self._budgets), outer_graph, random_state=seed)

        for budget in self._budgets:
            component_budgets = CC.component_budgets[budget]
            query_indices = [self.__apply_query_model_to_component(seed, data, budget, sampling_model, CC, comp_label)
                             for comp_label in range(CC.n_components) if component_budgets[comp_label] != 0]

            query_indices = list(chain.from_iterable(query_indices))  # flatten the list

            score = NearestNeighborAccuracy(data, query_indices, oracle, metric=prediction_model).score

            query_model, graph_method, metric = sampling_model
            new_data_points.append([np.round(100 * score), time.time(), budget, find_connected_component,
                                   f'{graph_method}-{metric}', str(query_model), prediction_model])# todo fix time

        return new_data_points

    def _get_data_points(self, search_parameters):
        find_connected_components = search_parameters[2]

        if not find_connected_components:
            return self.__get_query_indices(search_parameters)
        else:
            return self.__get_query_indices_using_connected_components(search_parameters)

    def create_score_csv(self):
        search_grid = self._build_search_grid()
        model_names = ['Budget', 'Find Components', 'Graph Type', 'Sampling Model', 'Prediction Method']

        start_time = time.time()
        for data_generator in self._data_generators:
            print(f'Beginning calculations for data generator [{data_generator[1]}]...')

            # calculate and save the scores of all search_parameters in search_grid
            import multiprocessing
            scores = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
                delayed(self._get_data_points)(search_parameters)
                for search_parameters in search_grid
            )
            scores = [inner for outer in scores for inner in outer]
            score_df = pd.DataFrame(scores, columns=['Score', 'Computation Time']+model_names)
            score_df.to_csv(f'results/[{data_generator[1]}]_scores.csv', index=False)

            # save the summary scores of all search_parameters in search_grid
            grouped_score_df = score_df[['Score', 'Computation Time']+model_names].groupby(model_names)
            average_score_df = grouped_score_df[['Score', 'Computation Time']].mean()
            average_score_df['Standard Deviation'] = grouped_score_df[['Score']].std()
            average_score_df[model_names] = grouped_score_df[model_names].first()
            average_score_df.to_csv(f'results/[{data_generator[1]}]_average_scores.csv', index=False)

        print('Total Run Time', time.time() - start_time)


class SummaryPlot:
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)

        self._name_dictionary = {
            'euclidean': 'Euclidean',
            '1fermat': 'Graph Metric',
            '2fermat': 'Fermat Metric',
            None: 'None',
            'False': 'None',
            'epsilon': 'Epsilon Graph',
            'knn': 'kNN Graph',
        }
        self._query_model_parameters = [
            ('Random', None, None),
            ('Kmeans', None, 'euclidean'),
            ('Kmeans', 'full', '1fermat'),
            ('Kmeans', 'full', '2fermat'),
            ('ProbCover', 'epsilon', 'euclidean'),
            ('ProbCover', 'epsilon', '1fermat'),
            ('ProbCover', 'epsilon', '2fermat')
        ]

        self.create_summary_plot(file_name)

    def _get_sub_df(self, query_model, graph_type, prediction_model):
        if query_model == 'Kmeans':
            sub_df = self.df[
                ((self.df["Sampling Model"] == f"<class 'QueryModels.{query_model}Sampling'>") |
                 (self.df["Sampling Model"] == f"<class 'QueryModels.Graph{query_model}Sampling'>")) &
                (self.df["Graph Type"] == graph_type) &
                (self.df["Prediction Method"] == prediction_model)
                ]
        else:
            sub_df = self.df[
                (self.df["Sampling Model"] == f"<class 'QueryModels.{query_model}Sampling'>") &
                (self.df["Graph Type"] == graph_type) &
                (self.df["Prediction Method"] == prediction_model)
                ]

        return sub_df

    def create_summary_plot(self, file_name):
        import re

        SPACING = 0.01
        fig, axes = plt.subplots(nrows=len(self._query_model_parameters), ncols=len(self._query_model_parameters[0]), figsize=(12, 22))
        fig.tight_layout(rect=(4*SPACING, SPACING, 1, 1 - 4*SPACING))
        fig.text(0.5, 1 - 2*SPACING, f"{re.split(r'[\[\]]', file_name)[1].upper()}",
                 va='center', fontsize=20, fontweight='bold', rotation='horizontal')
        fig.subplots_adjust(wspace=0.25, hspace=0.5)

        for i, parameters in enumerate(self._query_model_parameters):
            query_model = parameters[0]
            graph_type = f'{parameters[1]}-{parameters[2]}'

            pos = axes[i, 0].get_position()
            fig.text(SPACING, (pos.y0 + pos.y1) / 2, f"{query_model} ({self._name_dictionary[parameters[2]]})",
                     va='center', fontsize=12, fontweight='bold', rotation='vertical')

            for j, prediction_model in enumerate(['euclidean', '1fermat', '2fermat']):
                ax = axes[i, j]

                pos = ax.get_position()
                fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.5*SPACING, f'1NN ({self._name_dictionary[prediction_model]})',
                         ha='center', fontsize=12, fontweight='bold')

                sub_df = self._get_sub_df(query_model, graph_type, prediction_model)

                for k, find_component in enumerate(np.unique(sub_df["Find Components"])):
                    colors = ['red', 'green', 'blue']
                    plot_line = sub_df[sub_df["Find Components"] == find_component]

                    ax.plot(plot_line['Budget'], plot_line['Score'],
                            marker='o', linestyle='--', color=colors[k], label=self._name_dictionary[find_component], alpha=0.9)

                ax.grid(True)
                ax.legend()
                ax.set_xlabel('Size of Labeled Set')
                ax.set_ylabel('Accuracy (%)')
                # ax.set_ylim(95, 100.5)
                ax.set_ylim(50, 100)

        line = plt.Line2D([0.05, 0.95], [6 / 7, 6 / 7], color='black', linewidth=3, transform=fig.transFigure)
        fig.add_artist(line)
        line = plt.Line2D([0.05, 0.95], [3 / 7, 3 / 7], color='black', linewidth=3, transform=fig.transFigure)
        fig.add_artist(line)

        fig.show()



