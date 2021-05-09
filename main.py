import time
from read_data.read_data import read_data_adj_matrix
#from ant_colony_optimization.acs import AntColonySystem
#from ant_colony_optimization.acs_2 import AntColonySystem
from ant_colony_optimization.acs_3 import AntColonySystem

if __name__ == "__main__":

    # Alte fisiere

    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/old/queen5_5.col")
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/old/miles500.col")
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/old/miles750.col")
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/old/miles1000.col")

    # acs - 1 iter: 107 sec; 10 iter: ? sec ; ? best -
    # acs3 - 10 iter: 16 sec ; ? best - 29 gasit
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/old/school1_nsh.col")

    # acs - 1 iter: 19 sec; 10 iter: 187 sec ; 49 best - 49 gasit
    # acs3 - 10 iter: 6.1 sec ; 49 best - 49 gasit
    vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/old/zeroin.i.1.col")

    # Fisiere testare

    # acs - 10 iter: 0.07 sec ; 4 best - 4 gasit
    # acs3 - 10 iter: 0.04 sec ; 4 best - 4 gasit
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/myciel3.col")

    # acs - 10 iter: 140.9 sec ; 8 best - 8 gasit
    # acs3 - 10 iter: 4.8 sec ; 8 best - 8 gasit
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/myciel7.col")

    # acs - 10 iter: 96.8 sec ; 13 best - 16/17 gasit
    # acs3 - 10 iter: 4.1 sec ; 13 best - 19/20 gasit
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/queen13_13.col")

    # acs - 10 iter: 132.8 sec ; 31 best - 31 gasit
    # acs3 - 10 iter: 4.6 sec ; 31 best - 31 gasit
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/mulsol.i.5.col")

    # acs - 1 iter: 187 sec; 2 iter: 367 sec; 10 iter: 1779 sec ; 15 best - 16 gasit
    # acs3 - 10 iter: 25.9 sec ; 15 best - 17 gasit
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/le450_15b.col")

    # acs - 10 iter: ? ; 65 best - ? gasit
    # acs3 - 10 iter: 33.5 sec ; 65 best - 65 gasit
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/fpsol2.i.1.col")

    # acs - 1 iter: 504 sec; 10 iter: ? ; 31 best - ? gasit
    # acs3 - 10 iter: 53 sec ; 31 best - 31 gasit
    #vertices_count, edges_count, adjacency_matrix = read_data_adj_matrix(filepath="data_files/inithx.i.2.col")

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

    #acs_2 = AntColonySystem(number_vertices=vertices_count,
    #                      adjacency_matrix=adjacency_matrix,
    #                      number_ants=10,
    #                      number_iterations=10,
    #                      r=0.9,
    #                      evaporation_coefficient=0.8)
    #
    #acs_2.execute()

    acs_3 = AntColonySystem(number_vertices=vertices_count,
                            adjacency_matrix=adjacency_matrix,
                            number_ants=10,
                            number_iterations=10,
                            a=1,
                            b=1,
                            r=0.5)

    acs_3.execute()
