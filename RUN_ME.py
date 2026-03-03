from _compile_ import *
from CreateData import *


if __name__ == '__main__':
    # todo: check for duplicate situations
    # todo: look at label impurity withing a component


    data_generators = [(get_USPS_data, 'USPS'), (get_MNIST_data, 'MNIST'), (get_Satellite_data, 'satellite'),
                       (get_OPTDIGITS_data, 'digits'), (get_CIFAR10_data, 'CIFAR')]  # (generator, file_name)
    CompileResults(data_generators)
    # for data in data_generators:
    #     SummaryPlot(f"results/[{data[1]}]_average_scores.csv")
    #     ComparisonPlot(f"results/[{data[1]}]_average_scores.csv")
