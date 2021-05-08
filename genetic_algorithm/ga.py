import numpy as np

from genetic_algorithm.logging.compute_log_stats import compute_log_stats_realtime, compute_final_log_stats_realtime
from genetic_algorithm.logging.logger import GALogger


class GeneticAlgorithmEncodedPermutations:
    """
    Genetic algorithm using a specific encoding of permutations as representation.

    """

    def __init__(self,
                 vertecies_count,
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
                 evaluation_function=None,
                 evaluation_args=None,
                 evaluation_kwargs=None,
                 fitness_function=None,
                 fitness_args=None,
                 fitness_kwargs=None,
                 logging_path=None):

        self.pop_size = pop_size
        self.population = []

        self.vertecies_count = vertecies_count
        self.edges_count = edges_count
        self.adjacency_matrix = adj_matrix

        # This parameter is set as a parameter for the "execute" function
        self.iterations = 0

        # Set functions
        self.mutation = mutation_function
        self.crossover = crossover_function
        self.selection = selection_function
        self.evaluation = evaluation_function
        self.fitness = fitness_function

        # Set arguments
        self.mutation_args = mutation_args if mutation_args is not None else []
        self.crossover_args = crossover_args if crossover_args is not None else []
        self.selection_args = selection_args if selection_args is not None else []
        self.evaluation_args = evaluation_args if evaluation_args is not None else []
        self.fitness_args = fitness_args if fitness_args is not None else []

        # Set kw arguments
        self.mutation_kwargs = mutation_kwargs if mutation_kwargs is not None else {}
        self.crossover_kwargs = crossover_kwargs if crossover_kwargs is not None else {}
        self.selection_kwargs = selection_kwargs if selection_kwargs is not None else {}
        self.evaluation_kwargs = evaluation_kwargs if evaluation_kwargs is not None else {}
        self.fitness_kwargs = fitness_kwargs if fitness_kwargs is not None else {}

        self.evaluation_values = np.array([])
        self.fitness_values = np.array([])

        # Initialize the population
        self.init_pop()

        self.logger = GALogger(logging_path)

    def init_pop(self):
        # self.population = np.array([np.random.permutation(self.vertecies_count) for _ in range(self.pop_size)])
        self.population = np.array([
            [np.random.randint(0, remaining_vertecies_index)
             for remaining_vertecies_index in range(self.vertecies_count, 1, -1)]
            for chromosome_index in range(self.pop_size)
        ])

    def eval_population(self):
        return [self.evaluation(chromosome, self.adjacency_matrix,
                                *self.evaluation_args, **self.evaluation_kwargs)
                for chromosome in self.population]

    def execute(self, iterations=100, init=False):

        if init:
            self.init_pop()

        min_iteration = self.iterations
        max_iteration = self.iterations + iterations

        global_min = self.vertecies_count
        for iteration in range(min_iteration, max_iteration):
            print(f"Iteration {iteration} : ", end="")
            # Set the number of iterations completed (which is the same as the index of current iteration)
            self.iterations = iteration

            # Compute evaluation values
            self.evaluation_values = self.eval_population()

            # Print current min eval
            curr_min = np.min(self.evaluation_values)
            print(curr_min)
            if global_min > curr_min:
                global_min = curr_min

            # Compute fitness values
            self.fitness_values = self.fitness(self.evaluation_values,
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
            self.logger.log_generation(self.population, self.evaluation_values, self.fitness_values)
            compute_log_stats_realtime(self.logger.get_log_path(), iteration,
                                       self.evaluation_values, self.fitness_values)

        compute_final_log_stats_realtime(self.logger.get_log_path())
        print(global_min)
