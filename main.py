import time
from read_data.read_data import read_data_adj_matrix
from ant_colony_optimization.acs import AntColonySystem
from ant_colony_optimization.acs_2 import AntColonySystem

if __name__ == "__main__":

    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="queen5_5.col")
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="miles500.col")
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="miles750.col")
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="miles1000.col")

    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="school1_nsh.col")  # 1 iter - 107 sec
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="zeroin.i.1.col")  # 1 iter - 19 sec  # 49 gasit

    vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="myciel3.col")
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="queen13_13.col")
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="le450_15b.col")  # 1 iter - 187 sec  # 16 gasit
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="inithx.i.2.col")  # 1 iter -504 sec

    print(vertices_count)
    print(edges_count)
    print(adjacency_matrix)
    print()

    #acs = AntColonySystem(number_vertices=vertices_count,
    #                      adjacency_matrix=adjacency_matrix,
    #                      number_ants=10,
    #                      number_iterations=10,
    #                      a=1,
    #                      b=3,
    #                      r=0.9,
    #                      evaporation_coefficient=0.8)
    #
    #acs.execute()

    acs_2 = AntColonySystem(number_vertices=vertices_count,
                          adjacency_matrix=adjacency_matrix,
                          number_ants=10,
                          number_iterations=10,
                          r=0.9,
                          evaporation_coefficient=0.8)

    acs_2.execute()
