import numpy as np


def decode_to_permutation(chromosome):
    decoded = []
    current_vertecies = np.arange(chromosome.shape[0] + 1)

    for index in range(chromosome.shape[0]):
        decoded += [current_vertecies[chromosome[index]]]

        current_vertecies = np.delete(current_vertecies, chromosome[index])

    # Add the last vertex
    decoded += [current_vertecies[0]]

    return np.array(decoded)
