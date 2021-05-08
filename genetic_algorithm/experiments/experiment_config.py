
class ExperimentConfig:

    def __init__(self, log_path,
                 number_of_runs,
                 iterations,
                 pop_size,
                 evaluation_function, evaluation_kwargs,
                 fitness_function, fitness_kwargs,
                 selection_function, selection_kwargs,
                 crossover_function, crossover_kwargs,
                 mutation_function, mutation_kwargs,
                 ):

        self.log_path = log_path
        self.number_of_runs = number_of_runs
        self.iterations = iterations
        self.pop_size = pop_size

        self.evaluation_function = evaluation_function
        self.evaluation_kwargs = evaluation_kwargs

        self.fitness_function = fitness_function
        self.fitness_kwargs = fitness_kwargs

        self.selection_function = selection_function
        self.selection_kwargs = selection_kwargs

        self.crossover_function = crossover_function
        self.crossover_kwargs = crossover_kwargs

        self.mutation_function = mutation_function
        self.mutation_kwargs = mutation_kwargs

    def log_config(self):
        config_path = self.log_path[:-4] + "_config" + self.log_path[-4:]

        with open(config_path, "w") as file:
            file.write(f"Number of runs: {self.number_of_runs}\n"+
                       f"Iterations: {self.iterations}\n" +
                       f"Population size: {self.pop_size}\n" +

                       f"Evaluation function: {self.evaluation_function.__name__}\n" +
                       f"Evaluation function arguments: {self.evaluation_kwargs}\n" +

                       f"Fitness function: {self.fitness_function.__name__}\n" +
                       f"Fitness function arguments: {self.fitness_kwargs}\n" +

                       f"Selection function: {self.selection_function.__name__}\n" +
                       f"Selection function arguments: {self.selection_kwargs}\n" +

                       f"Crossover function: {self.crossover_function.__name__}\n" +
                       f"Crossover function arguments: {self.crossover_kwargs}\n" +

                       f"Mutation function: {self.mutation_function.__name__}\n" +
                       f"Mutation function arguments: {self.mutation_kwargs}\n"
                       )
