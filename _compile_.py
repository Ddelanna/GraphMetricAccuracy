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


class ComputeResults:
    def __init__(self, data_generator, search_parameters):
        self._budgets = [10, 20, 30, 40, 50, 75, 100]
        self._num_points = None  # num_points = None gets the entire data set

        self.data, self.oracle = data_generator(self._num_points)
        self.radius = BestParameter(self.data).best_radius(alpha=0.95)
        self.seed, self.find_connected_components, query_model_parameters, self.prediction_model = search_parameters
        (self.query_model, self.graph_method, self.metric) = query_model_parameters

    def __build_graph(self, data, graph_method, metric):
        if graph_method == 'full':
            return AdjacencyMatrices().distance_matrix(data, metric=metric)
        elif graph_method == 'epsilon':
            return AdjacencyMatrices().epsilon_graph(data, radius=self.radius, metric=metric)
        else:
            return None

    def __apply_query_model_to_component(self, budget, CC, component_label):
        component_data = self.data[CC.component_labels == component_label]
        component_budget = CC.component_budgets[budget][component_label]
        component_graph = self.__build_graph(component_data, self.graph_method, self.metric)
        query_indices = self.query_model(component_data, component_budget, component_graph, random_state=self.seed).query_indices
        return query_indices

    def _compute_data_points_using_standard_method(self):
        start = time.time()
        new_data_points = []

        graph = self.__build_graph(self.data, self.graph_method, self.metric)
        query_indices = self.query_model(self.data, max(self._budgets), graph, random_state=self.seed).query_indices
        end = time.time()

        for budget in self._budgets:
            score = NearestNeighborAccuracy(self.data, query_indices[:budget], self.oracle, metric=self.prediction_model).score

            new_data_points.append([np.round(100 * score), start-end, budget, self.find_connected_components,
                                    f'{self.graph_method}-{self.metric}', str(self.query_model), self.prediction_model])

        return new_data_points

    def _compute_data_points_using_connected_components(self):
        start = time.time()
        new_data_points = []

        outer_graph = self.__build_graph(self.data, graph_method=self.find_connected_components, metric='euclidean')
        CC = FindConnectedComponents(self.data, max(self._budgets), outer_graph, random_state=self.seed)
        end = time.time()

        for budget in self._budgets:
            component_budgets = CC.component_budgets[budget]
            query_indices = [self.__apply_query_model_to_component(budget, CC, comp_label)
                             for comp_label in range(CC.n_components) if component_budgets[comp_label] != 0]
            query_indices = list(chain.from_iterable(query_indices))  # flatten the list

            score = NearestNeighborAccuracy(self.data, query_indices, self.oracle, metric=self.prediction_model).score

            new_data_points.append([np.round(100 * score), start-end, budget, self.find_connected_components,
                                   f'{self.graph_method}-{self.metric}', str(self.query_model), self.prediction_model])# todo fix time

        return new_data_points

    def compute_data_points(self):
        if self.find_connected_components == False:
            return self._compute_data_points_using_standard_method()
        else:
            return self._compute_data_points_using_connected_components()


class CompileResults:
    def __init__(self, data_generators):
        self._num_iters = 10 # number of runs to average the score over
        self._data_generators = data_generators

        self.create_score_csv()

    def _build_search_grid(self):
        from itertools import product

        seeds = np.arange(self._num_iters)
        find_connected_components = ['epsilon', False]
        query_model_parameters = [(RandomSampling, None, None), # (query_model, graph_method, metric)
                                (KmeansSampling, None, 'euclidean'),
                                (GraphKmeansSampling, 'full', '2fermat'),
                                (ProbCoverSampling, 'epsilon', 'euclidean'),
                                (ProbCoverSampling, 'epsilon', '1fermat'),
                                (ProbCoverSampling, 'epsilon', '2fermat')]
        prediction_models = ['euclidean', '1fermat', '2fermat']
        # find_connected_components = ['epsilon']
        # query_model_parameters = [
        #                           (ProbCoverSampling, 'epsilon', '2fermat')]
        # prediction_models = ['2fermat']

        search_grid = product(seeds, find_connected_components, query_model_parameters, prediction_models)
        return list(search_grid)

    def create_score_csv(self):
        search_grid = self._build_search_grid()
        model_names = ['Budget', 'Find Components', 'Graph Type', 'Sampling Model', 'Prediction Method']

        start_time = time.time()
        for data_generator in self._data_generators:
            print(f'Beginning calculations for data generator [{data_generator[1]}]...')

            # calculate and save the scores of all search_parameters in search_grid
            import multiprocessing
            scores = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
                delayed(ComputeResults(data_generator[0], search_parameters).compute_data_points)()
                for search_parameters in search_grid
            )
            scores = [inner for outer in scores for inner in outer] # flatten the list by one level
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
            'False': 'Control',
            'epsilon': 'with Connected Components',
            'knn': 'kNN Graph',
        }
        self._query_model_parameters = [
            ('Random', None, None),
            ('ProbCover', 'epsilon', 'euclidean'),
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
        fig, axes = plt.subplots(nrows=len(self._query_model_parameters), ncols=2, figsize=(9, 12))
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

            for j, prediction_model in enumerate(['euclidean', '2fermat']):
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
                ax.set_ylim(50, 101)

        # line = plt.Line2D([0.05, 0.95], [6 / 7, 6 / 7], color='black', linewidth=3, transform=fig.transFigure)
        # fig.add_artist(line)
        # line = plt.Line2D([0.05, 0.95], [3 / 7, 3 / 7], color='black', linewidth=3, transform=fig.transFigure)
        # fig.add_artist(line)

        fig.show()


class ComparisonPlot:
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)

        self._name_dictionary = {
            'euclidean': 'Euclidean',
            '1fermat': 'Graph Metric',
            '2fermat': 'Fermat Metric',
            None: 'None',
            'False': 'Control',
            'epsilon': 'with Connected Components',
            'knn': 'kNN Graph',
        }
        self._query_model_parameters = [
            ('Random', 'False', f'None-None',  'euclidean'),
            ('ProbCover', 'False', 'epsilon-euclidean', 'euclidean'),
            ('Kmeans', 'False', f'None-euclidean', 'euclidean'),
            ('ProbCover', 'epsilon', 'epsilon-1fermat', '1fermat'),
            ('ProbCover', 'epsilon', 'epsilon-2fermat', '2fermat')
        ]

        self.create_comparison_plot(file_name)

    def create_comparison_plot(self, file_name):
        colors = ['red', 'green', 'black', 'blue', 'purple']
        labels = ['Random', 'ProbCover', 'Kmeans', '1Fermat ProbCover', '2Fermat ProbCover']
        for i, (query_model, use_CCS, graph_type, prediction_model) in enumerate(self._query_model_parameters):
            sub_df = self.df[
                (self.df["Sampling Model"] == f"<class 'QueryModels.{query_model}Sampling'>") &
                (self.df["Graph Type"] == graph_type) &
                (self.df["Prediction Method"] == prediction_model) &
                (self.df["Find Components"] == use_CCS)
                ]
            print(sub_df)
            plt.plot( sub_df['Budget'],  sub_df['Score'],
                    marker='o', linestyle='--', color=colors[i], label=labels[i], alpha=0.9)

        plt.grid(True)
        plt.legend()
        plt.xlabel('Size of Labeled Set')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 101)
        plt.title(file_name)
        plt.show()
