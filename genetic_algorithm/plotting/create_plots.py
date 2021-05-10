import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def process_data(generations_path):
    # generations_path = log_path[:-4] + "_generations_stats" + log_path[-4:]
    # runs_path = log_path[:-4] + "_runs_stats" + log_path[-4:]

    sep = "::"

    min_evals = []
    min_fitnesses = []

    with open(generations_path, "r") as file:
        generation_stats = file.readline()
        current_min_evals = []
        current_min_fitnesses = []

        while generation_stats:

            if "|||" in generation_stats:
                min_evals += [current_min_evals]
                min_fitnesses += [current_min_fitnesses]

                current_min_evals = []
                current_min_fitnesses = []

            else:
                gen_stats = generation_stats.split("|")[1]

                str_stats = gen_stats.split(sep)

                # Go through the stats and store the mins for eval and fitness
                for stats in str_stats:
                    split_stats = stats.split(":")
                    if split_stats[0] == "eval_min":
                        current_min_evals += [int(split_stats[1])]
                    elif split_stats[0] == "fitness_min":
                        current_min_fitnesses += [float(split_stats[1])]

            generation_stats = file.readline()

    global_min = 1e9
    global_min_run_index = 0
    for run_index, run in enumerate(min_evals):
        for iteration in run:
            if iteration < global_min:
                global_min = iteration
                global_min_run_index = run_index

    print(global_min)
    print(global_min_run_index)
    print(np.array(min_evals).shape)
    return global_min_run_index, global_min, min_evals[global_min_run_index]


def create_plots(generations_log_paths):
    PLOTS_PATH = ".\\plots"

    for title, gen_path in generations_log_paths:
        log_name = gen_path.split("/")[3]

        run_index, min_val, min_vals = process_data(gen_path)

        data_df = np.array([np.arange(len(min_vals))]).reshape(len(min_vals), 1)
        data_df = np.append(data_df, np.array([min_vals]).reshape(len(min_vals), 1), axis=1)

        df = pd.DataFrame(data=data_df,
                          columns=["Iterations", "Best Chromosome Minimum Color"])

        figure = plt.figure(figsize=(5.1, 4.1))
        sns.set_style("darkgrid")
        # sns.scatterplot(x=np.arange(len(min_vals))[::10], y=min_vals[::10], hue=df.loc[::10, "All Samples Covered"], s=150)
        # sns.lineplot(x=np.arange(len(min_vals))[::5], y=min_vals[::5], color='orange')
        sns.lineplot(x=np.arange(len(min_vals))[::5], y=min_vals[::5])
        plt.xlabel("Iterations", fontsize=10)
        plt.ylabel("Minimum Colors Count", fontsize=10)
        plt.tick_params(labelsize=9)
        plt.title(title, fontsize=14)
        plt.savefig(os.path.join(PLOTS_PATH, log_name + "_plot.png"))
        # plt.show()


if __name__ == '__main__':
    log_data = r"C:\Users\alexd\PycharmProjects\nim_tema_4_ga\genetic_algorithm\logging_results\experiments_results\myciel7\log_file_1620517618.5245903_generations_stats.txt"
    # process_data(log_data)

    # logs = [("fpsol2.i.1.col (496 vertecies, 11654 edges), 65-min-coloring",
    #          "../logging_results/experiments_results/fpsol2.i.1/log_file_1620514296.397173_generations_stats.txt"),
    #
    #         ("inithx.i.2.col (645 vertecies, 13979 edges), 31-min-coloring",
    #          "../logging_results/experiments_results/inithx.i.2/log_file_1620514231.0150423_generations_stats.txt"),
    #
    #         ("le450_15b.col (450 vertecies, 8169 edges), 15-min-coloring",
    #          "../logging_results/experiments_results/le450_15b/log_file_1620514193.4046335_generations_stats.txt"),
    #
    #         ("mulsol.i.5.col (186 vertecies, 3973 edges), 31-min-coloring",
    #          "../logging_results/experiments_results/mulsol.i.5/log_file_1620517704.466422_generations_stats.txt"),
    #
    #         ("myciel3.col(11 vertecies, 20 edges), 4-min-coloring",
    #          "../logging_results/experiments_results/myciel3/log_file_1620564501.4263854_generations_stats.txt"),
    #
    #         ("myciel7.col (191 vertecies, 2360 edges), 8-min-coloring",
    #          "../logging_results/experiments_results/myciel7/log_file_1620517618.5245903_generations_stats.txt"),
    #
    #         ("queen13_13.col (169 vertecies, 6656 edges), 13-min-coloring",
    #          "../logging_results/experiments_results/queen13_13/log_file_1620517639.1300583_generations_stats.txt"),
    #         ]


    title = "Genetic Algorithm"
    logs = [(title,
             "../logging_results/experiments_results/fpsol2.i.1/log_file_1620514296.397173_generations_stats.txt"),

            (title,
             "../logging_results/experiments_results/inithx.i.2/log_file_1620514231.0150423_generations_stats.txt"),

            (title,
             "../logging_results/experiments_results/le450_15b/log_file_1620514193.4046335_generations_stats.txt"),

            (title,
             "../logging_results/experiments_results/mulsol.i.5/log_file_1620517704.466422_generations_stats.txt"),

            (title,
             "../logging_results/experiments_results/myciel3/log_file_1620564501.4263854_generations_stats.txt"),

            (title,
             "../logging_results/experiments_results/myciel7/log_file_1620517618.5245903_generations_stats.txt"),

            (title,
             "../logging_results/experiments_results/queen13_13/log_file_1620517639.1300583_generations_stats.txt"),
            ]

    create_plots(logs)
