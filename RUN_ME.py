from _compile_ import SummaryResults, SummaryPlot
from CreateData import *


if __name__ == '__main__':
    # todo: check for duplicate situations
    # todo: look at label impurity withing a component
    # todo: make the algorithm suggest points to label in order
    # todo: rewrite BestParameter to take in a metric


    data_generators = [(get_swissroll_data, 'swissroll'), (get_smileyface_data, 'face'), (get_multmoons_data, 'moons')]  # (generator, file_name)
    SummaryResults(data_generators)
    for data in data_generators:
        SummaryPlot(f"results/[{data[1]}]_average_scores.csv")
