import numpy as np


def read_data_adj_matrix(filepath):

    """
    Read data as an adjacency matrix.
    :param filepath: the path to the file.
    :return: number of vertices, number of edges and the adjacency matrix.
    """

    adjacency_matrix = None
    vertices_count = 0
    edges_count = 0

    with open(filepath, "r") as file:
        line = file.readline()
        while line:
            if line.startswith("p edge"):
                data_graph = line.split()
                vertices_count = int(data_graph[2])
                edges_count = int(data_graph[3])
                adjacency_matrix = np.zeros((vertices_count, vertices_count))

            if line.startswith("e"):
                edge = line.split()
                node_1, node_2 = int(edge[1]) - 1, int(edge[2]) - 1

                adjacency_matrix[node_1][node_2] = 1
                adjacency_matrix[node_2][node_1] = 1

            line = file.readline()

    return vertices_count, edges_count, adjacency_matrix