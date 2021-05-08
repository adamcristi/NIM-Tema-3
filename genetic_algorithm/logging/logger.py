import time


class GALogger:

    def __init__(self, log_path):

        if log_path is None:
            self.log_path = f"./log_file_{time.time()}.txt"
        else:
            self.log_path = log_path

    def get_log_path(self):
        return self.log_path

    def get_generation_stats_log_path(self):
        return self.log_path[:-4] + "_generations_stats" + self.log_path[-4:]

    def get_runs_stats_log_path(self):
        return self.log_path[:-4] + "_runs_stats" + self.log_path[-4:]

    def log_generation(self, population, evaluation_values, fitness_values):

        with open(self.log_path, "a") as file:
            pop_list = population.tolist()

            for index in range(len(pop_list)):
                pop_list[index] = " ".join(map(str, pop_list[index]))

            file.write(":".join(pop_list))
            file.write(":::")
            file.write(" ".join(map(str, evaluation_values)))
            file.write(":::")
            file.write(" ".join(map(str, fitness_values)))
            file.write("\n")

    def log_run_end(self):
        with open(self.log_path, "a") as file:
            file.write("|||\n")

        with open(self.get_generation_stats_log_path(), "a") as file:
            file.write("|||\n")
