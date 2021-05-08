import numpy as np


def get_stats(arr):
    return np.min(arr), np.max(arr), np.mean(arr), np.std(arr)


def compute_log_stats_realtime(log_path, iteration, evaluation_values, fitness_values):
    new_log_path = log_path[:-4] + "_stats" + log_path[-4:]
    sep = "::"

    min_eval, max_eval, mean_eval, std_eval = get_stats(evaluation_values)
    min_fitness, max_fitness, mean_fitness, std_fitness = get_stats(fitness_values)

    with open(new_log_path, "a") as new_file:
        new_file.write(f"{iteration}|" +
                       f"eval_min:{min_eval}{sep}" +
                       f"eval_max:{max_eval}{sep}" +
                       f"eval_mean:{mean_eval}{sep}" +
                       f"eval_std:{std_eval}{sep}" +
                       f"fitness_min:{min_fitness}{sep}" +
                       f"fitness_max:{max_fitness}{sep}" +
                       f"fitness_mean:{mean_fitness}{sep}" +
                       f"fitness_std:{std_fitness}\n")


def compute_final_log_stats_realtime(log_path):
    sep = "::"

    stats_log_path = log_path[:-4] + "_stats" + log_path[-4:]

    min_evals = []
    min_fitnesses = []

    with open(stats_log_path, "r") as file:
        generation_stats = file.readline()

        while generation_stats:
            gen_stats = generation_stats.split("|")[1]

            str_stats = gen_stats.split(sep)

            # Go through the stats and store the mins for eval and fitness
            for stats in str_stats:
                split_stats = stats.split(":")
                if split_stats[0] == "eval_min":
                    min_evals += [int(split_stats[1])]
                elif split_stats[0] == "fitness_min":
                    min_fitnesses += [float(split_stats[1])]

            generation_stats = file.readline()

    min_evals_min, min_evals_max, min_evals_mean, min_evals_std = get_stats(min_evals)
    min_fitnesses_min, min_fitnesses_max, min_fitnesses_mean, min_fitnesses_std = get_stats(min_fitnesses)

    with open(stats_log_path, "a") as new_file:
        new_file.write(f"GLOBAL|" +
                       f"eval_min:{min_evals_min}{sep}" +
                       f"eval_max:{min_evals_max}{sep}" +
                       f"eval_mean:{min_evals_mean}{sep}" +
                       f"eval_std:{min_evals_std}{sep}" +
                       f"fitness_min:{min_fitnesses_min}{sep}" +
                       f"fitness_max:{min_fitnesses_max}{sep}" +
                       f"fitness_mean:{min_fitnesses_mean}{sep}" +
                       f"fitness_std:{min_fitnesses_std}\n")


# For log files that are already created and final
def compute_log_stats(log_path):
    new_log_path = log_path[:-4] + "_stats" + log_path[-4:]

    sep = "::"
    min_evals = []
    min_fitnesses = []

    with open(log_path, "r") as file:
        generation = file.readline()
        iteration = 0

        while generation:
            gen_list = generation.split(":::")

            chromosomes = [list(map(int, chrom.split(" "))) for chrom in gen_list[0].split(":")]
            evaluation_values = list(map(int, gen_list[1].split(" ")))
            fitness_values = list(map(float, gen_list[2].split(" ")))

            min_eval, max_eval, mean_eval, std_eval = get_stats(evaluation_values)
            min_fitness, max_fitness, mean_fitness, std_fitness = get_stats(fitness_values)

            min_evals += [min_eval]
            min_fitnesses += [min_fitness]

            with open(new_log_path, "a") as new_file:
                new_file.write(f"{iteration}:" +
                               f"eval_min:{min_eval}{sep}" +
                               f"eval_max:{max_eval}{sep}" +
                               f"eval_mean:{mean_eval}{sep}" +
                               f"eval_std:{std_eval}{sep}" +
                               f"fitness_min:{min_fitness}{sep}" +
                               f"fitness_max:{max_fitness}{sep}" +
                               f"fitness_mean:{mean_fitness}{sep}" +
                               f"fitness_std:{std_fitness}\n")

            iteration += 1
            generation = file.readline()

    # Last line is for global results (stats for the sample of the bests)
    min_evals_min, min_evals_max, min_evals_mean, min_evals_std = get_stats(min_evals)
    min_fitnesses_min, min_fitnesses_max, min_fitnesses_mean, min_fitnesses_std = get_stats(min_fitnesses)

    with open(new_log_path, "a") as new_file:
        new_file.write(f"GLOBAL:" +
                       f"eval_min:{min_evals_min}{sep}" +
                       f"eval_max:{min_evals_max}{sep}" +
                       f"eval_mean:{min_evals_mean}{sep}" +
                       f"eval_std:{min_evals_std}{sep}" +
                       f"fitness_min:{min_fitnesses_min}{sep}" +
                       f"fitness_max:{min_fitnesses_max}{sep}" +
                       f"fitness_mean:{min_fitnesses_mean}{sep}" +
                       f"fitness_std:{min_fitnesses_std}\n")
