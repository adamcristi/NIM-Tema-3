import numpy as np


def order_mutation(population, mutation_rate=0.3, mutation_choosing_rate=-1):
    pop_size, chromosome_size = population.shape

    if mutation_choosing_rate > 0:
        # Select chromosomes for mutation (save their indecies)
        indecies = np.nonzero(np.random.rand(pop_size) < mutation_choosing_rate)[0]
    else:
        # Apply mutation using all chromosomes in the process
        indecies = np.arange(0, pop_size)

    for index in indecies:
        # Create a mask with selected genes for mutation
        mutation_mask = np.array(np.random.rand(chromosome_size) < mutation_rate, dtype=np.byte)

        # Apply mutation
        population[index] = (population[index] + mutation_mask) % 2
