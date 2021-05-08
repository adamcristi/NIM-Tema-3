from evaluation.graph_coloring_evaluation import colors_count_eval_chromosome


# chromosome.shape[0] + 1 is the max coloring (aka the number of vertecies in the graph)
def eval_2(chromosome, adj_matrix):
    return 1 - colors_count_eval_chromosome(chromosome, adj_matrix) / (chromosome.shape[0] + 1)
