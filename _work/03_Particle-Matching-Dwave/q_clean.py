import numpy as np


def construct_Q(d, P):
    n = d.shape[0]
    Q = np.zeros((n ** 2, n ** 2))

    # Objective function
    for i in range(n):
        for j in range(n):
            # Q[n * i + j, n * i + j] -= 1 / (
            #             d[i, j] + 1)  # Multiply by -1 to convert the problem into a minimization problem.

            Q[n * i + j, n * i + j] += d[i, j]

    # Constraints
    for i in range(n):
        for j in range(n):
            for k in range(j + 1, n):
                # Constraint: each dot i should connect to exactly one cross.
                # Add penalty for selecting both cross j and cross k for dot i
                Q[n * i + j, n * i + k] += 2 * P
                Q[n * i + k, n * i + j] += 2 * P

            for k in range(i + 1, n):
                # Constraint: each cross j should connect to exactly one dot.
                # Add penalty for selecting both dot i and dot k for cross j
                Q[n * i + j, n * k + j] += 2 * P
                Q[n * k + j, n * i + j] += 2 * P

            # adding -P to the diagonal
            Q[n * i + j, n * i + j] -= 2 * P

    return Q
