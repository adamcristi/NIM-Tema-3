import numpy as np
import copy
import sys
import time

LOGS_PATH = "logs/"

class Ant:
    def __init__(self, nodes_graph, colours_nodes_graph, degrees_nodes_graph, adjacency_matrix, matrix_pheromones_trails,
                 a, b, r, start=None):

        self.nodes = copy.deepcopy(nodes_graph)
        self.colours_nodes = colours_nodes_graph
        self.degrees_nodes = degrees_nodes_graph
        self.matrix_adjacency = copy.deepcopy(adjacency_matrix)
        self.matrix_pheromones_trails = matrix_pheromones_trails
        self.alpha = a
        self.beta = b
        self.ro = r

        if start is None:
            self.current_node = np.random.choice(self.nodes)
        else:
            self.current_node = start

        self.visited_nodes = []
        self.unvisited_nodes = copy.deepcopy(self.nodes)

        self.colours_available = np.sort(copy.deepcopy(self.colours_nodes))
        self.colours_assigned = {node: None for node in self.nodes}
        #self.colours_counter = {colour: 0 for colour in self.colours_nodes}

        if len(self.visited_nodes) == 0:
            self.assign_colour(self.current_node, self.colours_nodes[0])

        self.num_colours_used = 0

        self.eval_colorings_nodes = 0

    def assign_colour(self, node, colour):
        self.colours_assigned[node] = colour
        #self.colours_counter[colour] += 1
        self.visited_nodes.append(node)
        self.unvisited_nodes.remove(node)

    def get_greedy_force_node(self, node):
        if self.degrees_nodes[node] == 0:
            return 0
        else:
            return 1 / self.degrees_nodes[node]

    def get_pheromone_trail_node(self, node, adjacency_node):
        return self.matrix_pheromones_trails[node, adjacency_node]

    def choose_next_node(self):
        if len(self.unvisited_nodes) == 0:
            return None
        elif len(self.unvisited_nodes) == 1:
            return self.unvisited_nodes[0]
        else:
            heuristic_values = []

            for possible_node in self.unvisited_nodes:
                greedy_force_node = self.get_greedy_force_node(node=possible_node)
                pheromone_trail_node = self.get_pheromone_trail_node(node=self.current_node,
                                                                     adjacency_node=possible_node)
                heuristic_values.append((greedy_force_node ** self.alpha) * (pheromone_trail_node ** self.beta))

            max_val_heuristic = np.max(heuristic_values)
            best_possible_nodes = []

            for index_node, node in enumerate(self.unvisited_nodes):
                if heuristic_values[index_node] >= max_val_heuristic:
                    best_possible_nodes.append(node)

            return np.random.choice(best_possible_nodes)

    def colorize(self):
        num_unvisited_nodes = len(self.unvisited_nodes)

        for index in range(num_unvisited_nodes):
            next_node = self.choose_next_node()
            tabu_colours = []

            for adj_node in range(self.matrix_adjacency.shape[0]):
                if self.matrix_adjacency[next_node][adj_node] == 1:
                    tabu_colours.append(self.colours_assigned[adj_node])

            for colour in self.colours_available:
                if colour not in tabu_colours:
                    self.assign_colour(node=next_node, colour=colour)
                    break

            self.current_node = next_node

        self.num_colours_used = len(set([value for value in self.colours_assigned.values() if value is not None]))

    def get_matrix_delta_pheromones_trails(self):
        matrix_delta_pheromones_trails = np.zeros(self.matrix_pheromones_trails.shape, dtype=np.float)
        for node_1 in self.nodes:
            if self.colours_assigned[node_1] is not None:
                matrix_delta_pheromones_trails[node_1, :] = (1 - self.ro) / self.num_colours_used

        return matrix_delta_pheromones_trails

    def local_search(self):
        for node, assigned_colour in self.colours_assigned.items():
            if assigned_colour == self.num_colours_used - 1:
                colours_adj_nodes = []
                for adj_node in range(self.matrix_adjacency.shape[0]):
                    if self.matrix_adjacency[node][adj_node] == 1:
                        colours_adj_nodes.append(self.colours_assigned[adj_node])

                for colour in range(self.num_colours_used - 1):
                    if colour not in colours_adj_nodes:
                        self.colours_assigned[node] = colour
                        break

        self.num_colours_used = len(set([value for value in self.colours_assigned.values() if value is not None]))

    def kempe_chain_interchange(self, first_node, second_node):
        colour_first_node = self.colours_assigned[first_node]
        colour_second_node = self.colours_assigned[second_node]
        update_colours_to_nodes = [first_node, second_node]
        index_check_node = 0

        while index_check_node < len(update_colours_to_nodes):
           current_node = update_colours_to_nodes[index_check_node]

           for adj_current_node in range(self.matrix_adjacency.shape[0]):
               if self.matrix_adjacency[current_node][adj_current_node] == 1:
                   if self.colours_assigned[adj_current_node] == colour_first_node or self.colours_assigned[adj_current_node] == colour_second_node:
                       if adj_current_node not in update_colours_to_nodes:
                           update_colours_to_nodes.append(adj_current_node)

           index_check_node += 1

        new_colours_assigned = copy.deepcopy(self.colours_assigned)
        if list(self.colours_assigned.values()).count(self.colours_assigned[first_node]) + list(self.colours_assigned.values()).count(self.colours_assigned[second_node]) != len(update_colours_to_nodes):
            for node in update_colours_to_nodes:
                if new_colours_assigned[node] == colour_first_node:
                    new_colours_assigned[node] = colour_second_node
                else:
                   new_colours_assigned[node] = colour_first_node

        return new_colours_assigned

    def kempe_chain_local_search(self):
        for node in self.nodes:
            self.eval_colorings_nodes += list(self.colours_assigned.values()).count(self.colours_assigned[node]) ** 2
        self.eval_colorings_nodes *= -1

        for node in self.nodes:
            stop = False
            for adj_node in self.nodes:
                if self.matrix_adjacency[node, adj_node] == 1:
                    new_colours_assigned = self.kempe_chain_interchange(first_node=node, second_node=adj_node)
                    new_eval_colorings_nodes = 0
                    for vertex in self.nodes:
                        new_eval_colorings_nodes += list(new_colours_assigned.values()).count(new_colours_assigned[vertex]) ** 2
                    new_eval_colorings_nodes *= -1

                    if new_eval_colorings_nodes < self.eval_colorings_nodes:
                        self.eval_colorings_nodes = new_eval_colorings_nodes
                        self.colours_assigned = new_colours_assigned
                        stop = True
                        break

            if stop is True:
                break

        self.num_colours_used = len(set([value for value in self.colours_assigned.values() if value is not None]))

        #nodes = []
        #evals_colorings_nodes = []
        #
        #for node in self.nodes:
        #    for adj_node in self.nodes:
        #        if self.matrix_adjacency[node, adj_node] == 1:
        #            nodes.append((node, adj_node))
        #            eval_colorings = (-1) * (self.colours_counter[self.colours_assigned[node]] ** 2 +
        #                             self.colours_counter[self.colours_assigned[adj_node]] ** 2)
        #            evals_colorings_nodes.append(eval_colorings)
        #
        #max_eval_colorings = np.min(evals_colorings_nodes)
        #index_nodes = np.random.choice(np.where(evals_colorings_nodes == max_eval_colorings)[0])
        #first_node = nodes[index_nodes][0]
        #second_node = nodes[index_nodes][1]
        #
        ##index_first_node = np.random.choice(len(self.visited_nodes))
        ##first_node = self.visited_nodes[index_first_node]
        #colour_first_node = self.colours_assigned[first_node]
        #
        ##index_second_node = np.random.choice(np.where(self.matrix_adjacency[first_node, :] == 1)[0])
        ##second_node = self.visited_nodes[index_second_node]
        #colour_second_node = self.colours_assigned[second_node]
        #
        #update_colours_to_nodes = [first_node, second_node]
        #index_check_node = 0
        #
        #while index_check_node < len(update_colours_to_nodes):
        #    current_node = update_colours_to_nodes[index_check_node]
        #
        #    for adj_current_node in range(self.matrix_adjacency.shape[0]):
        #        if self.matrix_adjacency[current_node][adj_current_node] == 1:
        #            if self.colours_assigned[adj_current_node] == colour_first_node or self.colours_assigned[adj_current_node] == colour_second_node:
        #                if adj_current_node not in update_colours_to_nodes:
        #                    update_colours_to_nodes.append(adj_current_node)
        #
        #    index_check_node += 1
        #
        #for node in update_colours_to_nodes:
        #    if self.colours_assigned[node] == colour_first_node:
        #        self.colours_assigned[node] = colour_second_node
        #    else:
        #        self.colours_assigned[node] = colour_first_node
        #
        #self.num_colours_used = len(set([value for value in self.colours_assigned.values() if value is not None]))

class AntColonySystem:
    def __init__(self, name_dataset, number_runs, number_vertices, adjacency_matrix, number_ants, number_iterations, a, b, r):
        self.dataset_name = name_dataset
        self.num_runs = number_runs
        self.num_nodes = number_vertices
        self.matrix_adjacency = adjacency_matrix
        self.num_ants = number_ants
        self.num_iterations = number_iterations
        self.alpha = a
        self.beta = b
        self.ro = r

        self.nodes = [node for node in range(number_vertices)]

        self.degree_nodes = None

        self.colour_nodes = None
        self.initialize_colours_nodes()

        self.matrix_pheromones_trails = None
        self.initialize_pheromones_trails()

        self.ants_colony = None

        self.current_min_num_colours_used = None
        self.current_best_ant = None
        self.number_ants_to_find_min = None

        self.global_min_num_colours_used = None
        self.global_best_ant = None
        self.number_iterations_to_find_min = None

        self.global_minimums = []
        self.numbers_iteartions = []

    def initialize_colours_nodes(self):
        self.degree_nodes = np.sum(self.matrix_adjacency, axis=1)
        counter_maximum_degree_nodes = int(np.max(self.degree_nodes))
        self.colour_nodes = np.array([colour for colour in range(counter_maximum_degree_nodes)])

    def initialize_pheromones_trails(self):
        self.matrix_pheromones_trails = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float)
        self.matrix_pheromones_trails[:, :] = (self.num_nodes ** 2) / self.ro

    def create_colony(self):
        self.ants_colony = [Ant(nodes_graph=self.nodes,
                                colours_nodes_graph=self.colour_nodes,
                                degrees_nodes_graph=self.degree_nodes,
                                adjacency_matrix=self.matrix_adjacency,
                                matrix_pheromones_trails=self.matrix_pheromones_trails,
                                a=self.alpha,
                                b=self.beta,
                                r=self.ro) for _ in range(self.num_ants)]

    def apply_evaporation(self):
        self.matrix_pheromones_trails *= self.ro

    def apply_intensification(self):
        self.matrix_pheromones_trails += self.current_best_ant.get_matrix_delta_pheromones_trails()

        max_pheromone_trail = 1 / (1 - self.ro * self.current_best_ant.num_colours_used)
        min_pheromone_trail = 0.087 * max_pheromone_trail

        self.matrix_pheromones_trails = np.where(self.matrix_pheromones_trails > max_pheromone_trail, max_pheromone_trail, self.matrix_pheromones_trails)
        self.matrix_pheromones_trails = np.where(self.matrix_pheromones_trails < min_pheromone_trail, min_pheromone_trail, self.matrix_pheromones_trails)

    def get_current_min_number_colours_used_and_best_ant(self):
        self.current_min_num_colours_used = None
        self.current_best_ant = None
        self.number_ants_to_find_min = None

        for index_ant, ant in enumerate(self.ants_colony):
            if self.current_min_num_colours_used is None:
                self.current_min_num_colours_used = ant.num_colours_used
                self.current_best_ant = ant
                self.number_ants_to_find_min = index_ant
            elif ant.num_colours_used < self.current_min_num_colours_used:
                self.current_min_num_colours_used = ant.num_colours_used
                self.current_best_ant = ant
                self.number_ants_to_find_min = index_ant

########################################################################################################################
# Logging #

    def write_number_run(self, number_run):
        with open(LOGS_PATH + self.dataset_name[:-4] + "_iterations.txt", "a+") as file:
            if number_run == 0:
                file.write(f"Run {number_run} \n\n")
            else:
                file.write(f"\nRun {number_run} \n\n")

    def get_log_info(self, log_iteration=False):

        delimiter = " ;" + " " * 4

        info = f"global_minimum_number_colours_used = {self.global_min_num_colours_used}{delimiter}"
        info += f"number_iterations_to_find_global_minimum = {self.number_iterations_to_find_min}{delimiter}"
        if log_iteration is True:
            info += f"current_iteration_minimum_number_colours_used = {self.current_min_num_colours_used}{delimiter}"
            info += f"number_ants_to_find_current_iteration_minimum = {self.number_ants_to_find_min}{delimiter}"

        return info

    def write_log_info_iteration(self, number_iteration):
        info_iteration = f"Iteration {number_iteration}: "
        info_iteration += self.get_log_info(log_iteration=True)

        with open(LOGS_PATH + self.dataset_name[:-4] + "_iterations.txt", "a+") as file:
            file.write(info_iteration + "\n")

    def write_log_info_run(self, number_run):
        info_run = f"Run {number_run}: "
        info_run += self.get_log_info()

        with open(LOGS_PATH + self.dataset_name[:-4] + "_runs.txt", "a+") as file:
            file.write(info_run + "\n")

        self.global_minimums.append(self.global_min_num_colours_used)
        self.numbers_iteartions.append(self.number_iterations_to_find_min)

    def write_log_info_runs(self):
        info_runs = f"\n\nRuns: "

        delimiter = " ;" + " " * 4

        info_runs += f"min_coverage_global_best = {np.min(self.global_minimums)}{delimiter}"
        info_runs += f"max_coverage_global_best = {np.max(self.global_minimums)}{delimiter}"
        info_runs += f"mean_coverage_global_best = {np.mean(self.global_minimums)}{delimiter}"
        info_runs += f"std_coverage_global_best = {np.std(self.global_minimums)}{delimiter}"

        with open(LOGS_PATH + self.dataset_name[:-4] + "_runs.txt", "a+") as file:
            file.write(info_runs + "\n")

########################################################################################################################

    def execute(self):
        open(LOGS_PATH + self.dataset_name[:-4] + "_iterations.txt", 'w').close()
        open(LOGS_PATH + self.dataset_name[:-4] + "_runs.txt", 'w').close()

        for run in range(self.num_runs):
            if run == 0:
                print(f"Run {run}")
            else:
                print(f"\nRun {run}")

            self.write_number_run(run)

            if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                start = time.time_ns()
            else:
                start = time.time()

            self.matrix_pheromones_trails = None
            self.initialize_pheromones_trails()

            self.global_min_num_colours_used = None
            self.global_best_ant = None
            self.number_iterations_to_find_min = None

            for iteration in range(self.num_iterations):
                self.create_colony()

                for ant in self.ants_colony:
                    ant.colorize()

                for ant in self.ants_colony:
                    ant.local_search()
                    #ant.kempe_chain_local_search()

                self.get_current_min_number_colours_used_and_best_ant()

                if self.global_min_num_colours_used is None:
                    self.global_min_num_colours_used = self.current_min_num_colours_used
                    self.global_best_ant = self.current_best_ant
                    self.number_iterations_to_find_min = iteration
                elif self.current_min_num_colours_used < self.global_min_num_colours_used:
                    self.global_min_num_colours_used = self.current_min_num_colours_used
                    self.global_best_ant = self.current_best_ant
                    self.number_iterations_to_find_min = iteration

                self.apply_evaporation()
                self.apply_intensification()

                if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                    end = time.time_ns()
                    print(f"Iteration {iteration} - Elapsed time: {(end - start) / 1e9} seconds.")
                else:
                    end = time.time()
                    print(f"Iteration {iteration} - Elapsed time: {(end - start)} seconds.")

                self.write_log_info_iteration(iteration)

            self.write_log_info_run(run)

        self.write_log_info_runs()

            #print(self.global_min_num_colours_used)
            #print(self.global_best_ant.colours_assigned)
            # print(self.global_best_ant.unvisited_nodes)
            # print(self.global_best_ant.visited_nodes)
            # print(self.global_best_ant.colours_nodes)


