import time

import CreateData
import HelperFunctions
import PredictionMethods
import SamplingAlgorithms
import temp
from HelperFunctions import AdjacencyMatrices
from temp import *




if __name__ == '__main__':
    # todo: fix AdjacencyMatrices
    # todo: record runtimes for creating distance / adjacency matrices and for the algorithm


    data, oracle = CreateData.create_spiral_data(10000)
    dist1 = AdjacencyMatrices().sparse_distance_matrix(data, 1.5)
    dist2 = AdjacencyMatrices().binary_epsilon_graph(data, 1.5)

    time1 = time.time()
    print(SamplingAlgorithms.ConnectedComponentSampling(data, 6, dist1, random_state=1).query_indices)
    time2 = time.time()

    print(SamplingAlgorithms.ConnectedComponentSampling(data, 6, dist2, random_state=1).query_indices)
    time3 = time.time()
    print(time3-time2, time2-time1)


