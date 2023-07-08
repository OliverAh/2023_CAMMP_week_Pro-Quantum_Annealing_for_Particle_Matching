import numpy as np
from qbsolv import solution_slicer, QBSolve_quantum_solution, Q_dict, QBSolve_classical_solution
from qbsolv2 import QBSolve_classical_solution, QBSolve_quantum_solution
from dist import calc_phi_ij, find_argmax
from access_test import bqm_run
import scipy
from q import Q
from misc import sanity_check, index_sort, counter, classical, quantum, plot_hist, correct_solution_counter
from q_clean import construct_Q
from ortools import solve_quadratic_binary
import time
from kmeans import q_cluster

# data

p1 = np.loadtxt('coords_30p0s_03_10.txt')
p2 = np.loadtxt('coords_32p5s_03_10.txt')
# p1 = np.array([[1.5,1],[1.5,2],[1.5,3]])
# p2 = np.array([[1,1],[1,2],[1,3]])

# # distance matrix
dist = calc_phi_ij(p1, p2)

#
# # hungarian method
# s = time.time()
row_ind, hung_sol = scipy.optimize.linear_sum_assignment(dist)  # col_ind is the assignment of p2 to p1
# e = time.time()
print("Big Hungarian row_ind: ", row_ind)
print("Big Hungarian Solution: ", hung_sol)
#
# # Q matrix
# l = np.amax(dist) * 1.01  # lambda
# q = Q(dist, l, inverse=False, full_q=False)
# # q = construct_Q(dist, l)
# # np.savetxt('Q.txt', q, delimiter=',')
# q_dic = Q_dict(q)

q_list, dot_indices_list, cross_indices_list = q_cluster(p1, p2, 5)

for q, dot_indices, cross_indices in zip(q_list, dot_indices_list, cross_indices_list):

    dist = calc_phi_ij(p1[dot_indices], p2[cross_indices])

    # hungarian method
    row_ind, hung_sol = scipy.optimize.linear_sum_assignment(dist)
    print("Hungarian row_ind: ", dot_indices)
    print("Hungarian Solution: ", hung_sol)

    # # Solve Q matrix (Quantum/ Classical)
    # # Gurobi
    # gurobi_sol = solve_quadratic_binary(q)
    # low_gurobi_sol = solve_quadratic_binary(q)['solution']
    # # print("Gurobi Solution: ", gurobi_sol)
    #
    #
    # classical_sol = classical(q)
    # print("% sim correct: ", correct_solution_counter(classical_sol, hung_sol))
    # plot_hist(classical_sol, hung_sol, title="Classical Solution")
    # low_classical_sol = classical_sol[0]['solution']
    # # print("classical solution: ", classical_sol)
    #
    #
    # quantum_sol = quantum(q)
    # print("% QC correct: ", correct_solution_counter(quantum_sol, hung_sol))
    # plot_hist(quantum_sol, hung_sol, title="Quantum Solution")
    # low_quantum_sol = quantum_sol[0]['solution']
    # # print("quantum solution: ", quantum_sol[0:4])



# calculate the solution distance
# print("Hung Dist", np.sum(dist[row_ind, hung_sol]))
# print("Gurobi Dist", np.sum(dist[row_ind, low_gurobi_sol]))
# print("Classical Dist", np.sum(dist[row_ind, low_classical_sol]))
# print("Quantum Dist", np.sum(dist[row_ind, low_quantum_sol]))


