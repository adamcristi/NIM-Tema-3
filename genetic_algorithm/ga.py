import numpy as np


class GeneticAlgorithmEncodedPermutations:
    """
    Genetic algorithm using a specific encoding of permutations as representation.

    """

    def __init__(self,
                 nodes_count,
                 edges_count,
                 adj_matrix,
                 pop_size=100,
                 mutation_function=None,
                 mutation_args=None,
                 mutation_kwargs=None,
                 crossover_function=None,
                 crossover_args=None,
                 crossover_kwargs=None,
                 selection_function=None,
                 selection_args=None,
                 selection_kwargs=None,
                 fitness_function=None,
                 fitness_args=None,
                 fitness_kwargs=None
                 ):

        self.pop_size = pop_size
        self.population = []

        self.nodes_count = nodes_count
        self.edges_count = edges_count
        self.adjacency_matrix = adj_matrix

        # This parameter is set as a parameter for the "execute" function
        self.iterations = 0

        # Set functions
        self.mutation = mutation_function
        self.crossover = crossover_function
        self.selection = selection_function
        self.fitness = fitness_function

        # Set arguments
        self.mutation_args = mutation_args
        self.crossover_args = crossover_args
        self.selection_args = selection_args
        self.fitness_args = fitness_args

        # Set kw arguments
        self.mutation_kwargs = mutation_kwargs
        self.crossover_kwargs = crossover_kwargs
        self.selection_kwargs = selection_kwargs
        self.fitness_kwargs = fitness_kwargs

        self.fitness_values = np.array([])

        # Initialize the population
        self.init_pop()

    def init_pop(self):
        # self.population = np.array([np.random.permutation(self.nodes_count) for _ in range(self.pop_size)])
        self.population = np.array([
            [np.random.randint(0, remaining_nodes_index) for remaining_nodes_index in range(self.nodes_count, 1, -1)]
            for chromosome_index in range(self.pop_size)
        ])

    def execute(self, iterations=100, init=False):

        if init:
            self.init_pop()

        min_iteration = self.iterations
        max_iteration = self.iterations + iterations

        for iteration in range(min_iteration, max_iteration):
            # Set the number of iterations completed (which is the same as the index of current iteration)
            self.iterations = iteration

            # Compute fitness values
            self.fitness_values = self.fitness(self.population,
                                               *self.fitness_args, **self.fitness_kwargs)
            # Apply selection
            self.population = self.selection(self.population, self.fitness_values,
                                             *self.selection_args, **self.selection_kwargs)
            # Apply crossover
            self.population = self.crossover(self.population,
                                             *self.crossover_args, **self.crossover_kwargs)
            # Apply mutation
            self.population = self.mutation(self.population,
                                            *self.mutation_args, **self.mutation_kwargs)
            # Logging
            # ....................
