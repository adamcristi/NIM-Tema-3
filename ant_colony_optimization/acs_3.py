import numpy as np
import copy
import sys
import time


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

        if len(self.visited_nodes) == 0:
            self.assign_colour(self.current_node, self.colours_nodes[0])

        self.num_colours_used = 0

    def assign_colour(self, node, colour):
        self.colours_assigned[node] = colour
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

        self.num_colours_used = len(set(self.colours_assigned.values()))

    def get_matrix_delta_pheromones_trails(self):
        matrix_delta_pheromones_trails = np.zeros(self.matrix_pheromones_trails.shape, dtype=np.float)
        for node_1 in self.nodes:
            if self.colours_assigned[node_1] is not None:
                matrix_delta_pheromones_trails[node_1, :] = (1 - self.ro) / self.num_colours_used

        return matrix_delta_pheromones_trails


class AntColonySystem:
    def __init__(self, number_vertices, adjacency_matrix, number_ants, number_iterations, a, b, r):
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

        self.global_min_num_colours_used = None
        self.global_best_ant = None

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

        for ant in self.ants_colony:
            if self.current_min_num_colours_used is None:
                self.current_min_num_colours_used = ant.num_colours_used
                self.current_best_ant = ant
            elif ant.num_colours_used < self.current_min_num_colours_used:
                self.current_min_num_colours_used = ant.num_colours_used
                self.current_best_ant = ant

    def execute(self):

        if sys.version_info.major == 3 and sys.version_info.minor >= 7:
            start = time.time_ns()
        else:
            start = time.time()

        for iteration in range(self.num_iterations):
            self.create_colony()

            for ant in self.ants_colony:
                ant.colorize()

            self.get_current_min_number_colours_used_and_best_ant()

            if self.global_min_num_colours_used is None:
                self.global_min_num_colours_used = self.current_min_num_colours_used
                self.global_best_ant = self.current_best_ant
            elif self.current_min_num_colours_used < self.global_min_num_colours_used:
                self.global_min_num_colours_used = self.current_min_num_colours_used
                self.global_best_ant = self.current_best_ant

            self.apply_evaporation()
            self.apply_intensification()

            if sys.version_info.major == 3 and sys.version_info.minor >= 7:
                end = time.time_ns()
                print(f"Iteration {iteration} - Elapsed time: {(end - start) / 1e9} seconds.")
            else:
                end = time.time()
                print(f"Iteration {iteration} - Elapsed time: {(end - start)} seconds.")

        print(self.global_min_num_colours_used)
        print(self.global_best_ant.colours_assigned)
        # print(self.global_best_ant.unvisited_nodes)
        # print(self.global_best_ant.visited_nodes)
        # print(self.global_best_ant.colours_nodes)


