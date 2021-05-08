import numpy as np
import copy
import sys
import time


class Ant:
    def __init__(self, nodes_graph, colours_nodes_graph, adjacency_matrix, matrix_pheromones_trails, start=None):

        self.nodes = copy.deepcopy(nodes_graph)
        self.colours_nodes = colours_nodes_graph
        self.matrix_adjacency = copy.deepcopy(adjacency_matrix)
        self.matrix_pheromones_trails = matrix_pheromones_trails

        if start is None:
            self.current_node = np.random.choice(self.nodes)
        else:
            self.current_node = start

        self.visited_nodes = []
        self.unvisited_nodes = copy.deepcopy(self.nodes)

        self.colours_available = np.sort(copy.deepcopy(self.colours_nodes))
        self.colours_assigned = {node: None for node in self.nodes}
        self.colours_counter = {colour: 0 for colour in self.colours_nodes}

        if len(self.visited_nodes) == 0:
            self.assign_colour(self.current_node, self.colours_nodes[0])

        self.num_colours_used = 0

    def assign_colour(self, node, colour):
        self.colours_assigned[node] = colour
        self.colours_counter[colour] += 1
        self.visited_nodes.append(node)
        self.unvisited_nodes.remove(node)

    def get_greedy_force_node(self, node, colour):
        num_nodes_uncolored_by_move = 0
        for adj_node in range(self.matrix_adjacency.shape[0]):
            if self.matrix_adjacency[node][adj_node] == 1 and colour == self.colours_assigned[adj_node]:
                num_nodes_uncolored_by_move += 1
        return num_nodes_uncolored_by_move

    def get_pheromone_trail_node(self, node, adjacency_node):
        return self.matrix_pheromones_trails[node, adjacency_node]

    def choose_next_node_and_colour(self):
        if len(self.unvisited_nodes) == 0:
            return None, None
        else:
            greedy_force_values_nodes = np.zeros((len(self.unvisited_nodes), len(self.colours_nodes)))
            pheromone_trail_values_nodes = []
            for index_node, possible_node in enumerate(self.unvisited_nodes):
                pheromone_trail_values_nodes.append(self.get_pheromone_trail_node(node=self.current_node, adjacency_node=possible_node))
                for index_colour, possible_colour in enumerate(self.colours_nodes):
                    greedy_force_node = self.get_greedy_force_node(node=possible_node, colour=possible_colour)
                    greedy_force_values_nodes[index_node, index_colour] = greedy_force_node

            print(greedy_force_values_nodes.shape)
            print(greedy_force_values_nodes)

            max_greedy_force_value = np.max(greedy_force_values_nodes)
            max_pheromone_trail_value = np.max(pheromone_trail_values_nodes)

            best_possible_moves = []
            for index_node, node in enumerate(self.unvisited_nodes):
                if pheromone_trail_values_nodes[index_node] >= max_pheromone_trail_value:
                    indexes_colours = np.where(greedy_force_values_nodes[index_node, :] == max_greedy_force_value)[0]
                    moves = [(node, self.colours_nodes[index_colour]) for index_colour in indexes_colours]
                    best_possible_moves.extend(moves)

            if len(best_possible_moves) > 1:
                index_best_move = np.random.choice([index for index in range(len(best_possible_moves))])
            else:
                index_best_move = 0

            return best_possible_moves[index_best_move]

    def colorize(self):
        num_unvisited_nodes = len(self.unvisited_nodes)

        for index in range(num_unvisited_nodes):
            #print(self.choose_next_node_and_colour())
            next_node, next_node_colour = self.choose_next_node_and_colour()

            self.assign_colour(node=next_node, colour=next_node_colour)

            self.num_colours_used = len(set(self.colours_assigned.values()))
            self.current_node = next_node

    def get_matrix_delta_pheromones_trails(self):
        matrix_delta_pheromones_trails = np.zeros(self.matrix_pheromones_trails.shape, dtype=np.float)
        for node_1 in self.nodes:
            for node_2 in self.nodes:
                if self.colours_assigned[node_1] == self.colours_assigned[node_2]:
                    matrix_delta_pheromones_trails[node_1][node_2] = self.colours_counter[self.colours_assigned[node_1]] ** 2

        return matrix_delta_pheromones_trails


class AntColonySystem:
    def __init__(self, number_vertices, adjacency_matrix, number_ants, number_iterations, r, evaporation_coefficient):
        self.num_nodes = number_vertices
        self.matrix_adjacency = adjacency_matrix
        self.num_ants = number_ants
        self.num_iterations = number_iterations
        self.ro = r
        self.coef_evap = evaporation_coefficient

        self.nodes = [node for node in range(number_vertices)]

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
        counter_maximum_degree_nodes = int(np.max(np.sum(self.matrix_adjacency, axis=1)))
        self.colour_nodes = np.array([colour for colour in range(counter_maximum_degree_nodes)])

    def initialize_pheromones_trails(self):
        self.matrix_pheromones_trails = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float)

    def create_colony(self):
        self.ants_colony = [Ant(nodes_graph=self.nodes,
                                colours_nodes_graph=self.colour_nodes,
                                adjacency_matrix=self.matrix_adjacency,
                                matrix_pheromones_trails=self.matrix_pheromones_trails) for _ in range(self.num_ants)]

    def apply_evaporation(self):
        self.matrix_pheromones_trails *= (1 - self.coef_evap)

    def apply_intensification(self):
        self.matrix_pheromones_trails = self.ro * self.matrix_pheromones_trails + self.current_best_ant.get_matrix_delta_pheromones_trails()

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

            print(self.ants_colony[0].colours_nodes)

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


