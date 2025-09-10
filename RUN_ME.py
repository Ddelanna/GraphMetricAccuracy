import time
import CreateData
import HelperFunctions
import PredictionMethods
import SamplingAlgorithms
from HelperFunctions import AdjacencyMatrices
import graphlearning as gl
from _temporary_ import *
from _compile_ import Iterate

if __name__ == '__main__':
    # completed: made all methods able to take in sparse/dense and a fixed seed
    # completed? verified the models
    # completed: improved run time for kmeans
    # completed: compilation of results
    # completed: record the run times
    # completed: reinstated method to find radius
    # completed: made compilation function parallel
    # in progress: updating the documentation

    # todo: ask about GraphMetricAccuracy graph method
    # todo: for kmeans, the distance is always euclidean
    # todo: for knn, the distance is always euclidean
    # todo: radius is always computed regardless if it is needed or not
    # todo: check if other processes can be made parallel?
    # todo: how to find best knn value?

    # data, oracle = CreateData.get_MNIST_data(1000, random_state=1)
    # dist_mat = AdjacencyMatrices.distance_matrix(data)
    #
    # time1 = time.time()
    # _, query_indices = adaptive_sampling(np.array(data), oracle, dist_mat, 10, random_state=1)
    # time2 = time.time()
    # query_indices2 = SamplingAlgorithms.KmeansSampling(data, 10, gl.graph(dist_mat), random_state=1).query_indices
    # time3 = time.time()
    # accuracy = PredictionMethods.GraphMetricAccuracy(data, query_indices2, oracle, radius=0.25).score
    # accuracy2 = PredictionMethods.EuclideanAccuracy(data, query_indices2, oracle, radius=0.25).score
    #
    # print('Accuracy:', accuracy, accuracy2)
    # print('old', time2-time1, 'new', time3-time2)

    Iterate()



