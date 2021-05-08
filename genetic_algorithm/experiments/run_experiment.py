import time

from genetic_algorithm.ga import GeneticAlgorithmEncodedPermutations
from genetic_algorithm.logging.compute_log_stats import compute_final_log_stats
from genetic_algorithm.logging.logger import GALogger
from read_data.read_data import read_data_adj_matrix


def run_experiment(data_file_path, experiment_config):

    vertecies_count, edges_count, adj_matrix = read_data_adj_matrix(data_file_path)

    log_path = experiment_config.log_path

    logger = GALogger(log_path)
    generation_stats_log_path = logger.get_generation_stats_log_path()
    runs_stats_log_path = logger.get_runs_stats_log_path()

    experiment_config.log_config()

    for run_index in range(experiment_config.number_of_runs):
        print(f"Run : {run_index}")
        ga = GeneticAlgorithmEncodedPermutations(vertecies_count, edges_count, adj_matrix,
                                                 pop_size=experiment_config.pop_size,
                                                 evaluation_function=experiment_config.evaluation_function,
                                                 evaluation_kwargs=experiment_config.evaluation_kwargs,
                                                 fitness_function=experiment_config.fitness_function,
                                                 fitness_kwargs=experiment_config.fitness_kwargs,
                                                 selection_function=experiment_config.selection_function,
                                                 selection_kwargs=experiment_config.selection_kwargs,
                                                 crossover_function=experiment_config.crossover_function,
                                                 crossover_kwargs=experiment_config.crossover_kwargs,
                                                 mutation_function=experiment_config.mutation_function,
                                                 mutation_kwargs=experiment_config.mutation_kwargs,
                                                 logger=logger,
                                                 )

        start = time.time()
        ga.execute(iterations=experiment_config.iterations)
        end = time.time()
        print(f"Time elapsed : {(end - start)}")

    compute_final_log_stats(generation_stats_log_path, runs_stats_log_path)
