from evaluation.decode_to_permutation import decode_to_permutation


def are_adjacent(vertex_1, vertex_2, adj_matrix):
    return vertex_2 in adj_matrix[vertex_1]


def colors_count_eval_chromosome(chromosome, adj_matrix):
    decoded = decode_to_permutation(chromosome)

    colors_count = 0
    start_index = 0

    for index in range(0, decoded.shape[0]):
        for check_index in range(start_index, index):
            if are_adjacent(decoded[check_index], decoded[index], adj_matrix):
                # Count the color of the previous interval
                colors_count += 1
                start_index = index
                break

    # Count the last color
    colors_count += 1

    return colors_count
