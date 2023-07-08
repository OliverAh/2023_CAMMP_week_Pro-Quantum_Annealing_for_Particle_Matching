import cvxpy as cp
import numpy as np
from misc import solution_slicer, find_argmax
from gurobipy import Model, GRB
import gurobipy as gp
import time



def solve_quadratic_binary(Q, print_time=True):
    with gp.Env(empty=True) as env:
        env.setParam("LICENSEID", 0)
        env.setParam("OutputFlag", 0)
        env.start()

        # Number of variables
        n = len(Q)

        # Create a new model
        m = Model("qubo")

        # Create variables
        x = m.addVars(n, vtype=GRB.BINARY, name="x")

        # Set objective
        quad_expr = 0
        for i in range(n):
            for j in range(n):
                quad_expr += Q[i][j] * x[i] * x[j]
        m.setObjective(quad_expr, GRB.MINIMIZE)

        # Optimize model
        if print_time:
            start = time.time()
        m.optimize()
        if print_time:
            end = time.time()
            print("Time taken by Gurobi: ", end - start)

    # Return the optimized binary values
    solution = []
    for v in m.getVars():
        solution.append(v.x)

    # slicing the list of solutions into a list of lists
    num_particles = int(np.sqrt(len(Q)))
    sliced_solutions = solution_slicer(solution, num_particles)

    # finding the argmax of the sliced solutions
    max_outputs = find_argmax(sliced_solutions)

    # Each sample is a dictionary where the keys are the variable labels and the values are the variable values
    solution = {
        'solution': max_outputs,
        'energy': None,
        'num_occurrences': None,
        'optimal_value': m.objVal
    }

    return solution


# Example usage
# Q = np.array([[2, -1, 0], [-1, 3, -1], [0, -1, 2]])  # Quadratic matrix
# x_optimal = solve_quadratic_binary(Q)
#
# print("Optimal x:", x_optimal)