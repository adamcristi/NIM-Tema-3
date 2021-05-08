
def fitness_2(evaluation_values, pressure):

    min_eval, max_eval = min(evaluation_values), max(evaluation_values)
    eval_diff = max_eval - min_eval + 1e-10

    return [(1.01 + (eval_val - min_eval) / eval_diff) ** pressure for eval_val in evaluation_values]
