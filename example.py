from SamplingHelperFunctions import KnnAccuracy, GraphMetricAccuracy
from SamplingAlgorithms import ProbCoverSampling, ConnectedComponentSampling
from CreateData import *



if __name__ == '__main__':

    unlabeled_points, oracle = create_spiral_data(1000, dimension=7)

    PC = ProbCoverSampling(unlabeled_points, budget=6, radius=1.5)
    print('KNN Accuracy', KnnAccuracy(PC, oracle, create_plots=True).score)

    CC = ConnectedComponentSampling(unlabeled_points, budget=6, alpha=0.95)
    print('GraphMetric Accuracy', GraphMetricAccuracy(CC, oracle, create_plots=True).score)
