from _compile_ import *
from CreateData import *


if __name__ == '__main__':
    # todo: check for duplicate situations
    # todo: look at label impurity withing a component
    # todo: compute all matrices up front before parallelized process
    # todo: possibly rewrite as serial
    # todo: print out the number of connected components
    # todo: compare 1NN-fermat methods


    data_generators = [(get_smileyface_data, 'face'),]
                       # (get_COIL20_data, 'Coil20'),
                       # (get_USPS_data, 'USPS'),
                       # (get_MNIST_data, 'MNIST'),
                       # (get_swissroll_data, 'swissroll'),
                       # (get_OPTDIGITS_data, 'digits'),
                       # (get_Satellite_data, 'satellite'),
                       # ]  # (generator, file_name)
    CompileResults(data_generators, in_parallel=False)
    for data in data_generators:
        SummaryPlot(f"results/[{data[1]}]_average_scores.csv")
        ComparisonPlot(f"results/[{data[1]}]_average_scores.csv")
