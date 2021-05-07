import time

import numpy as np

from evaluation.decode_to_permutation import decode_to_permutation
from fitness.fitness_1 import fitness_1
from genetic_algorithm.ga import GeneticAlgorithmEncodedPermutations
from operators.selection.wheel_of_fortune import wheel_of_fortune_selection
from operators.crossover.double_cut_crossover import double_cut_crossover
from operators.crossover.single_cut_crossover import single_cut_crossover
from operators.mutation.add_one_mutation import add_one_mutation
from read_data.read_data import read_data_adj_matrix
from evaluation.graph_coloring_evaluation import colors_count_eval_chromosome

if __name__ == '__main__':
    # vertecies_count, edges_count, adj_matrix = read_data_adj_matrix("queen5_5.col")
    vertecies_count, edges_count, adj_matrix = read_data_adj_matrix("queen13_13.col")

    ga = GeneticAlgorithmEncodedPermutations(vertecies_count, edges_count, adj_matrix,
                                             pop_size=100,
                                             evaluation_function=colors_count_eval_chromosome,
                                             evaluation_kwargs={},
                                             fitness_function=fitness_1,
                                             fitness_kwargs={"pressure": 2},
                                             selection_function=wheel_of_fortune_selection,
                                             selection_kwargs={},
                                             crossover_function=double_cut_crossover,
                                             crossover_kwargs={"crossover_rate": 0.3},
                                             mutation_function=add_one_mutation,
                                             mutation_kwargs={"mutation_rate": 0.01},
                                             )

    start = time.time_ns()
    ga.execute(iterations=1000)
    end = time.time_ns()

    print((end - start) / 1e9)

    # # print(np.array([np.random.permutation(10) for x in range(10)]))
    #
    # # print(np.arange(0, 100))
    #
    # # print(np.random.permutation(10))
    #
    # # pop_size = 10
    # # chromosome_size = 10
    # #
    # # print(np.array([[np.random.randint(0, index)
    # #                  for index in range(chromosome_size, 1, -1)]
    # #                 for _ in range(pop_size)
    # #                 ]))
    # # print(np.random.randint(0, chromosome_size))
    #
    # # print(decode_to_permutation(np.array([0, 1, 2, 4, 3, 0, 1])))
    #
    # # print(22 in np.array([[10, 1, 2, 4, 2, 3, 1], [32, 32, 2, 545, 45, 3, 543], [324, 22, 4, 23, 234, 545, 45]])[2])
    #
    #
    # # chromosome = np.array([0, 2, 4, 7, 6, 1, 3, 5])
    # # chromosome = np.array([0, 2, 4, 7, 6, 1, 3, 5])
    #
    # # chromosome = np.array([0, 1, 2, 4, 3, 0, 1])
    #
    # chromosome = np.array([0, 2, 4, 7, 6, 1, 5, 3])
    #
    # adj_matrix = [[1, 7],
    #               [0, 2],
    #               [1, 3],
    #               [2, 4],
    #               [3, 5],
    #               [4, 6],
    #               [5, 7],
    #               [6, 0]]
    #
    # print(graph_coloring_eval_chromosome(chromosome, adj_matrix))
