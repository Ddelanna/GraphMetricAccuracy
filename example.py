import CreateData
import HelperFunctions
import PredictionMethods
import SamplingAlgorithms
from HelperFunctions import AdjacencyMatrices




class Iterate:
    def __init__(self, data_generation, sampling_method, prediction_method, random_state=None):
        NUM_POINTS = 1000
        BUDGET = 10
        RADIUS = 1.5

        unlabeled_points, oracle = data_generation(num_points=NUM_POINTS, random_state=random_state)

        for adjacency_matrix in [AdjacencyMatrices().binary_epsilon_graph(unlabeled_points, RADIUS)]:
            model = sampling_method(unlabeled_points, BUDGET, adjacency_matrix=adjacency_matrix, random_state=random_state)
            score = prediction_method(model, oracle).score
            print('score', score)

if __name__ == '__main__':
    # todo: metrics to use: euclidean, fermat (p=1,2,4?)
    # todo: fix AdjacencyMatrices
    # todo: switch sampling methods over to sparse / improve runtime

    # todo: using MNIST, FashionMNIST, and CIFAR from jeff's graphlearning github
    # todo: record runtimes for creating distance / adjacency matrices and for the algorithm

    Iterate(data_generation=CreateData.create_spiral_data,
            sampling_method=SamplingAlgorithms.ConnectedComponentSampling,
            prediction_method=PredictionMethods.GraphMetricAccuracy,
            random_state=1)

    distance_matrix = AdjacencyMatrices().distance_matrix(data)
    SamplingAlgorithms.KmeansSampling(data, budget, distance_matrix)





