import time

from evaluation.evaluation_functions.eval_1 import eval_1
from evaluation.evaluation_functions.eval_2 import eval_2
from fitness.fitness_1 import fitness_1
from fitness.fitness_2 import fitness_2
from genetic_algorithm.experiments.experiment_config import ExperimentConfig
from genetic_algorithm.operators.selection.wheel_of_fortune import wheel_of_fortune_selection
from genetic_algorithm.operators.crossover.double_cut_crossover import double_cut_crossover
from genetic_algorithm.operators.mutation.add_one_mutation import add_one_mutation
from genetic_algorithm.experiments.run_experiment import run_experiment

if __name__ == '__main__':
    data_file_path = "../data_files/queen5_5.col"

    experiment_config = ExperimentConfig(log_path=f"./logging_results/log_file_{time.time()}.txt",
                                         number_of_runs=30,
                                         iterations=1000,
                                         pop_size=100,
                                         evaluation_function=eval_1,
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
    run_experiment(data_file_path, experiment_config)

    #     ================================

    experiment_config = ExperimentConfig(log_path=f"./logging_results/log_file_{time.time()}.txt",
                                         number_of_runs=30,
                                         iterations=1000,
                                         pop_size=100,
                                         evaluation_function=eval_2,
                                         evaluation_kwargs={},
                                         fitness_function=fitness_2,
                                         fitness_kwargs={"pressure": 3},
                                         selection_function=wheel_of_fortune_selection,
                                         selection_kwargs={},
                                         crossover_function=double_cut_crossover,
                                         crossover_kwargs={"crossover_rate": 0.4},
                                         mutation_function=add_one_mutation,
                                         mutation_kwargs={"mutation_rate": 0.01},
                                         )
    run_experiment(data_file_path, experiment_config)
