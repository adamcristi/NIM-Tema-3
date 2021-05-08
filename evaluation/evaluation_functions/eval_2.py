import numpy as np

from evaluation.graph_coloring_evaluation import colors_count_eval_chromosome


# chromosome.shape[0] + 1 is the max coloring (aka the number of vertecies in the graph)
def eval_2(chromosome, adj_matrix):
    return 1 - colors_count_eval_chromosome(chromosome, adj_matrix) / (chromosome.shape[0] + 1)


def reverse_eval_2(eval_value, max_colorings_count, use_round=True):
    value = (1 - eval_value) * max_colorings_count
    return np.round(value) if use_round else value
