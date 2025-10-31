from _compile_ import SummaryResults, SummaryPlot


if __name__ == '__main__':
    # todo: check for duplicate situations
    # todo: look at label impurity withing a component
    # todo: make the algorithm suggest points to label in order
    # todo: rewrite BestParameter to take in a metric



    SummaryResults()
    for data in ['spiral']:
        SummaryPlot(f"results/[{data}]_average_scores.csv")
