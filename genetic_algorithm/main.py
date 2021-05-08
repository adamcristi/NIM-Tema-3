import time

from fitness.fitness_1 import fitness_1
from genetic_algorithm.ga import GeneticAlgorithmEncodedPermutations
from genetic_algorithm.operators.selection.wheel_of_fortune import wheel_of_fortune_selection
from genetic_algorithm.operators.crossover.double_cut_crossover import double_cut_crossover
from genetic_algorithm.operators.mutation.add_one_mutation import add_one_mutation
from read_data.read_data import read_data_adj_matrix
from evaluation.graph_coloring_evaluation import colors_count_eval_chromosome

if __name__ == '__main__':
    # vertecies_count, edges_count, adj_matrix = read_data_adj_matrix("queen5_5.col")
    # vertecies_count, edges_count, adj_matrix = read_data_adj_matrix("queen13_13.col")
    vertecies_count, edges_count, adj_matrix = read_data_adj_matrix("inithx.i.1.col")
    # vertecies_count, edges_count, adj_matrix = read_data_adj_matrix("miles1500.col")

    ga = GeneticAlgorithmEncodedPermutations(vertecies_count, edges_count, adj_matrix,
                                             pop_size=100,
                                             evaluation_function=colors_count_eval_chromosome,
                                             evaluation_kwargs={},
                                             fitness_function=fitness_1,
                                             fitness_kwargs={"pressure": 3},
                                             selection_function=wheel_of_fortune_selection,
                                             selection_kwargs={},
                                             crossover_function=double_cut_crossover,
                                             crossover_kwargs={"crossover_rate": 0.4},
                                             mutation_function=add_one_mutation,
                                             mutation_kwargs={"mutation_rate": 0.01},
                                             )

    start = time.time_ns()
    ga.execute(iterations=1000)
    end = time.time_ns()

    print((end - start) / 1e9)
    # tttt
