def read_data_adj_matrix(filepath):

    """
    Read data as an adjacency matrix.
    :param filepath: the path to the file.
    :return: number of vertecies, number of edges and the adjacency matrix.
    """

    adjacency_list = []
    vertecies_count = 0
    edges_count = 0

    with open(filepath, "r") as file:
        line = file.readline()
        while line:
            if line.startswith("p edge"):
                elems = line.split()
                vertecies_count = int(elems[2])
                edges_count = int(elems[3])
                adjacency_list = [list([]) for node_index in range(vertecies_count)]

            if line.startswith("e"):
                elems = line.split()
                node_1, node_2 = int(elems[1]) - 1, int(elems[2]) - 1

                adjacency_list[node_1] += [node_2]

            line = file.readline()

    return vertecies_count, edges_count, adjacency_list