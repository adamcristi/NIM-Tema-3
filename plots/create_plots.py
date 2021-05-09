import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib

matplotlib.use('TkAgg')
matplotlib.style.use('seaborn-darkgrid')
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker


#log_name = 'best_run_myciel3.txt'
#log_name = 'best_run_myciel7.txt'
log_name = 'best_run_queen13_13.txt'

# LOGS_PATH is a must to be the absolute path to the logs
LOGS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'plots')
PLOTS_PATH = os.path.join(os.path.split(os.path.abspath(os.getcwd()))[0], 'plots')


def filter_data(value):
    if value == '' or value == ';' or value == '=':
        return False
    else:
        return True


def create_plot():
    mins_num_global = []
    mins_num_current_iteration = []

    title_plot = log_name.split("_", 2)[2].split(".")[0]

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

    ax.plot(np.arange(len(mins_num_current_iteration)), mins_num_global, color='blue', linewidth=4)
    ax.plot(np.arange(len(mins_num_current_iteration)), mins_num_current_iteration, color='orange', linewidth=2)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Minimum Colours Count')

    ax.legend(["Minimum Global", "Minimum Iteration"], frameon=True)

    ax.set_xticks(np.arange(0, len(mins_num_global), step=1))
    ax.set_yticks(np.arange(np.min(mins_num_current_iteration), np.max(mins_num_current_iteration) + 1, step=1))

    #plt.setp(ax.get_xticklabels()[-1], visible=False)
    plt.title(title_plot)
    #plt.savefig(os.path.join(PLOTS_PATH, log_name + "_plot.png"))
    plt.show()



########################################################################################################################
# First plot #

    #ax1 = axs[0]
    ## sns.lineplot(x=np.arange(len(coverages_global_best)), y=coverages_global_best, color='orange', ax=ax1, linewidth=3.5)
    #line_1 = ax1.plot(np.arange(len(coverages_global_best)), coverages_global_best, color='orange', linewidth=3.5, label="Coverage")
    #
    #ax1.set_xlabel('Iterations')
    #ax1.set_ylabel('Coverage', color='orange')
    #ax1.tick_params(axis='y', labelcolor='orange')
    ## ax1.set_xticks(np.arange(0, len(coverages_global_best) + 1, step=len(coverages_global_best) / 5))
    #
    #step_yticks_ax1 = (np.max(coverages_global_best) - np.min(coverages_global_best)) / 10
    #ax1.set_yticks(np.around(np.arange(np.min(coverages_global_best), np.max(coverages_global_best) + step_yticks_ax1, step=step_yticks_ax1)))
    #
    #second_ax1 = ax1.twinx()
    ## sns.lineplot(x=np.arange(len(evaluation_values_global_best)), y=evaluation_values_global_best, color='blue', ax=second_ax1)
    #line_2 = second_ax1.plot(np.arange(len(evaluation_values_global_best)), evaluation_values_global_best, color='blue', label="Evaluation")
    #
    #second_ax1.lines[0].set_linestyle("--")
    #second_ax1.set_ylabel('Evaluation', color='blue') #, rotation=270, labelpad=10)
    #second_ax1.tick_params(axis='y', labelcolor='blue')
    #second_ax1.set_xticks(np.arange(0, len(evaluation_values_global_best) + 1, step=len(evaluation_values_global_best) / 5))
    #
    #step_yticks_second_ax1 = (np.max(evaluation_values_global_best) - np.min(evaluation_values_global_best)) / 11
    #second_ax1.set_yticks(np.around(np.arange(np.min(evaluation_values_global_best), np.max(evaluation_values_global_best) + step_yticks_second_ax1,
    #                                          step=step_yticks_second_ax1), decimals=5))
    #
    #lines_second_plot = line_1 + line_2
    #labels_lines_first_plot = [line.get_label() for line in lines_second_plot]
    #
    #second_ax1.legend(lines_second_plot, labels_lines_first_plot, loc=0, frameon=True)
    #second_ax1.set_title("Best Global Particle")
    #
#########################################################################################################################
# Se#cond plot #
    #
    #ax2 = axs[1]
    ## sns.lineplot(x=np.arange(len(coverages_best_iteration)), y=coverages_best_iteration, color='orange', ax=ax2, linewidth=3.5, label="Coverage Current Best")
    #line_3 = ax2.plot(np.arange(len(coverages_best_iteration)), coverages_best_iteration, color='orange', linewidth=3.5, label="Coverage")
    #
    #ax2.set_xlabel('Iterations')
    #ax2.set_ylabel('Coverage', color='orange')
    #ax2.tick_params(axis='y', labelcolor='orange')
    ## ax2.set_xticks(np.arange(0, len(coverages_best_iteration)+1, step=len(coverages_best_iteration) / 5))
    #
    #step_yticks_ax2 = (np.max(coverages_best_iteration) - np.min(coverages_best_iteration)) / 10
    #ax2.set_yticks(np.around(np.arange(np.min(coverages_best_iteration), np.max(coverages_best_iteration) + step_yticks_ax2, step=step_yticks_ax2)))
    #
    #second_ax2 = ax2.twinx()
    ## sns.lineplot(x=np.arange(len(evaluation_values_best_iteration)), y=evaluation_values_best_iteration, color='blue', ax=second_ax2, label="Evaluation Current Best")
    #line_4 = second_ax2.plot(np.arange(len(evaluation_values_best_iteration)), evaluation_values_best_iteration, color='blue', label="Evaluation")
    #
    #second_ax2.lines[0].set_linestyle("--")
    #second_ax2.set_ylabel('Evaluation', color='blue') #, rotation=270, labelpad=10)
    #second_ax2.tick_params(axis='y', labelcolor='blue')
    #second_ax2.set_xticks(np.around(np.arange(0, len(evaluation_values_best_iteration) + 1, step=len(evaluation_values_best_iteration) / 10)))
    #
    #step_yticks_second_ax2 = (np.max(evaluation_values_best_iteration) - np.min(evaluation_values_best_iteration)) / 15
    #second_ax2.set_yticks(np.around(np.arange(np.min(evaluation_values_best_iteration), np.max(evaluation_values_best_iteration) + step_yticks_second_ax2, step=step_yticks_second_ax2), decimals=10))
    #
    #lines_second_plot = line_3 + line_4
    #labels_lines_second_plot = [line.get_label() for line in lines_second_plot]
    #
    ##second_ax2.grid(False, axis='y')
    #second_ax2.legend(lines_second_plot, labels_lines_second_plot, loc=0, frameon=True)
    #second_ax2.set_title("Best Current Particle")
    #
    ##frame_legend_ax2 = legend_ax2.get_frame()
    ##frame_legend_ax2.set_facecolor('darkgray')
    #
    ##ax2.legend()
    ##second_ax2.legend()
    ##second_ax2.legend(labels=["Evaluation Current Best"])
    #
    ##tick_spacing = 1
    ##second_ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.suptitle(title_plot)
    #plt.savefig(os.path.join(PLOTS_PATH, log_name.split('.')[0] + "_plot.png"))
    ##plt.show()
    #
    #
    ###sns.lineplot(x=np.arange(len(min_vals))[::5], y=min_vals[::5], color='orange')
    ##sns.lineplot(x=np.arange(len(min_vals))[::5], y=min_vals[::5], hue=df.loc[::5, "All Samples Covered"])
    ##plt.xlabel("Iterations", fontsize=10)
    ##plt.ylabel("Best Chromosome Minimum Candidates", fontsize=10)
    ##plt.tick_params(labelsize=9)
    ##plt.title(type_eval_chromosome, fontsize=14)
    ##plt.savefig(os.path.join(PLOTS_PATH, log_name + "_plot.png"))


# Create plot
create_plot()