from CreateData import *
from HelperFunctions import BestParameter, AdjacencyMatrices


class oldBestParameter:
    def __init__(self, data, budget, metric='euclidean'):
        self.data = data
        self.budget = budget

        self.distance_matrix = AdjacencyMatrices().distance_matrix(data, metric=metric) # todo

    def _compute_coverage(self, radius):
        """ :return coverage: the ratio of points in the same connected component as a labeled point
            to the total number of points """

        adjacency_matrix = (self.distance_matrix <= radius).astype(int)
        from scipy.sparse.csgraph import connected_components
        n_components, component_labels = connected_components(csgraph=adjacency_matrix, directed=False, return_labels=True)
        _, component_sizes = np.unique(component_labels, return_counts=True)
        ordered_components_sizes = sorted(component_sizes, reverse=True)
        coverage = sum(ordered_components_sizes[:self.budget]) / self.data.shape[0]

        return coverage

    def best_radius(self, alpha):
        diameter = np.max(self.distance_matrix)
        tolerance = 0.01  # todo : how to determine tolerance?
        radius_lowerbound, radius_upperbound = 0, diameter

        while (radius_upperbound - radius_lowerbound) > tolerance:

            radius_to_compute = (radius_lowerbound + radius_upperbound) / 2

            coverage = self._compute_coverage(radius_to_compute)
            if coverage > alpha:
                radius_upperbound = radius_to_compute
            else:
                radius_lowerbound = radius_to_compute

        best_radius = (radius_lowerbound + radius_upperbound) / 2
        return best_radius

for data_generator in [get_USPS_data, get_Satellite_data]:
    data, oracle = data_generator(1000)
    print('\n', BestParameter(data).best_radius(0.975))

    for budget in [5, 10, 20, 30, 40]:
        print(budget, oldBestParameter(data, budget).best_radius(0.975))