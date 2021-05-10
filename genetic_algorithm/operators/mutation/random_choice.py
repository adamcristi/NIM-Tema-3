import numpy as np


def random_choice(population, mutation_rate=0.01, mutation_choosing_rate=-1):
    pop_size, chromosome_size = population.shape

    if mutation_choosing_rate > 0:
        # Select chromosomes for mutation (save their indecies)
        indecies = np.nonzero(np.random.rand(pop_size) < mutation_choosing_rate)[0]
    else:
        # Apply mutation using all chromosomes in the process
        indecies = np.arange(0, pop_size)

    for index in indecies:
        for gene_index in range(0, chromosome_size):
            if np.random.random() < mutation_rate:
                # Apply mutation
                new_val = np.random.randint(0, chromosome_size + 1 - gene_index)
                while new_val == population[index][gene_index]:
                    new_val = np.random.randint(0, chromosome_size + 1 - gene_index)
                population[index][gene_index] = new_val

    return population
