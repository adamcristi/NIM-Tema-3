import time

from evaluation.evaluation_functions.eval_1 import eval_1
from fitness.fitness_1 import fitness_1
from genetic_algorithm.experiments.experiment_config import ExperimentConfig
from genetic_algorithm.operators.selection.wheel_of_fortune import wheel_of_fortune_selection
from genetic_algorithm.operators.crossover.double_cut_crossover import double_cut_crossover
from genetic_algorithm.operators.mutation.add_one_mutation import add_one_mutation
from genetic_algorithm.experiments.run_experiment import run_experiment

if __name__ == '__main__':
    data_file_path = []
    # data_file_path += ["../data_files/myciel3.col"]  # myciel3.col (11,20), 4, MYC
    data_file_path += ["../data_files/myciel7.col"]  # myciel7.col (191,2360), 8, MYC
    # data_file_path += ["../data_files/queen13_13.col"]  # queen13_13.col (169,6656), 13, SGB
    # data_file_path += ["../data_files/mulsol.i.5.col"]  # mulsol.i.5.col (186,3973), 31, REG
    # data_file_path += ["../data_files/le450_15b.col"]  # le450_15b.col (450,8169), 15, LEI
    # data_file_path += ["../data_files/fpsol2.i.1.col"]  # fpsol2.i.1.col (496,11654), 65, REG
    # data_file_path += ["../data_files/inithx.i.2.col"]  # inithx.i.2.col (645, 13979), 31, REG

    """
    myciel3.col(11, 20), 4, MYC
    myciel7.col (191,2360), 8, MYC
    queen13_13.col (169,6656), 13, SGB
    mulsol.i.5.col (186,3973), 31, REG
    le450_15b.col (450,8169), 15, LEI
    fpsol2.i.1.col (496,11654), 65, REG
    inithx.i.2.col (645, 13979), 31, REG
    """

    for path in data_file_path:
        experiment_config = ExperimentConfig(log_path=f"./logging_results/log_file_{time.time()}.txt",
                                             number_of_runs=30,
                                             iterations=1000,
                                             pop_size=100,
                                             evaluation_function=eval_1,
                                             evaluation_kwargs={},
                                             fitness_function=fitness_1,
                                             fitness_kwargs={"pressure": 4},
                                             selection_function=wheel_of_fortune_selection,
                                             selection_kwargs={},
                                             crossover_function=double_cut_crossover,
                                             crossover_kwargs={"crossover_rate": 0.4},
                                             mutation_function=add_one_mutation,
                                             mutation_kwargs={"mutation_rate": 0.02},
                                             )
        run_experiment(path, experiment_config)
