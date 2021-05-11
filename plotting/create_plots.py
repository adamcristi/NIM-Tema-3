import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
matplotlib.style.use('seaborn-darkgrid')
import matplotlib.pyplot as plt


#log_name = 'best_run_myciel3.txt'
#log_name = 'best_run_myciel7.txt'
#log_name = 'best_run_queen13_13.txt'
#log_name = 'best_run_mulsol.i.5.txt'
#log_name = 'best_run_le450_15b.txt'
#log_name = 'best_run_fpsol2.i.1.txt'
log_name = 'best_run_inithx.i.2.txt'

# LOGS_PATH is a must to be the absolute path to the logs
LOGS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'plotting', 'logs_plots')
PLOTS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'plotting', 'plots')


def filter_data(value):
    if value == '' or value == ';' or value == '=':
        return False
    else:
        return True


def create_plot():
    mins_num_global = []
    mins_num_current_iteration = []

    #title_plot = log_name.split("_", 2)[2].split(".")[0]
    #title_plot = 'Hybrid Ant Colony Optimization'
    title_plot = 'Ant Colony System'

    with open(os.path.join(LOGS_PATH, log_name), "r") as fd:
        current_iteration = fd.readline().strip()

        while current_iteration:
            data_current_iteration = current_iteration.split(" ")
            data_current_iteration = list(filter(filter_data, data_current_iteration))
            data_current_iteration[0] = data_current_iteration[0].replace(':', '')

            current_min_num_global = int(data_current_iteration[data_current_iteration.index('global_minimum_number_colours_used') + 1])
            current_min_num_current_iteration = int(data_current_iteration[data_current_iteration.index('current_iteration_minimum_number_colours_used') + 1])

            mins_num_global.append(current_min_num_global)
            mins_num_current_iteration.append(current_min_num_current_iteration)

            current_iteration = fd.readline().strip()

    print(mins_num_global)
    print(mins_num_current_iteration)

    figure = plt.figure(figsize=(5.1, 4.1))
    ax = figure.add_subplot()

    ax.plot(np.arange(len(mins_num_current_iteration)), mins_num_global, color='tab:blue', linewidth=4)
    ax.plot(np.arange(len(mins_num_current_iteration)), mins_num_current_iteration, color='orange', linewidth=2)

    ax.lines[1].set_linestyle("--")

    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Minimum Colours Count', fontsize=12)

    ax.legend(["Minimum Global", "Minimum Iteration"], frameon=True)

    ax.set_xticks(np.arange(0, len(mins_num_global), step=1))
    ax.set_yticks(np.arange(np.min(mins_num_current_iteration), np.max(mins_num_current_iteration) + 1, step=1))

    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=11)

    #plt.setp(ax.get_xticklabels()[-1], visible=False)
    plt.title(title_plot, fontsize=14)
    plt.savefig(os.path.join(PLOTS_PATH, log_name + "_plot.png"))
    #plt.show()


# Create plot
create_plot()