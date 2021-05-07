import numpy as np

from genetic_algorithm.ga import GeneticAlgorithmEncodedPermutations
from operators.selection.wheel_of_fortune import wheel_of_fortune_selection
from operators.crossover import *
from operators.mutation import *
from read_data.read_data import read_data_adj_matrix

if __name__ == '__main__':
    # nodes_count, edges_count, adj_matrix = read_data_adj_matrix("queen5_5.col")

    # ga = GeneticAlgorithmEncodedPermutations(nodes_count, edges_count, adj_matrix,
    #                                                pop_size=100,
    #                                                fitness_function=None,
    #                                                fitness_kwargs={},
    #                                                selection_function=wheel_of_fortune_selection,
    #                                                selection_kwargs={},
    #                                                crossover_function=None,
    #                                                crossover_kwargs={},
    #                                                mutation_function=None,
    #                                                mutation_kwargs={},
    #                                                )

    # print(np.array([np.random.permutation(10) for x in range(10)]))

    # print(np.arange(0, 100))

    # print(np.random.permutation(10))

    pop_size = 10
    chromosome_size = 10

    print(np.array([[np.random.randint(0, index)
                     for index in range(chromosome_size, 1, -1)]
                    for _ in range(pop_size)
                    ]))
    # print(np.random.randint(0, chromosome_size))
