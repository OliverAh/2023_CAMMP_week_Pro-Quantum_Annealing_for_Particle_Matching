import numpy as np
from qbsolv import solution_slicer, QBSolve_quantum_solution, Q_dict, QBSolve_classical_solution
from qbsolv2 import QBSolve_classical_solution, QBSolve_quantum_solution
from dist import calc_phi_ij, find_argmax
from access_test import bqm_run
import scipy
from q import Q
from misc import sanity_check, index_sort, counter

# p1 = np.array([[1.5,1],[1.5,2],[1.5,3],[3,1],[3,2],[3,3]])
# p2 = np.array([[1,1],[1,2],[1,3],[2.5,1],[2.5,2],[2.5,3]])

# load the coordinate files
# p1 = np.loadtxt('coords_30p0s_04_100.txt')
# p2 = np.loadtxt('coords_32p5s_04_100.txt')
p1 = np.loadtxt('coords_30p0s_02_30.txt')
p2 = np.loadtxt('coords_32p5s_02_30.txt')


# p1 = np.array([[1.5,1],[1.5,2],[1.5,3]])
# p2 = np.array([[1,1],[1,2],[1,3]])

print("shape of p1: ", np.shape(p1))
print("shape of p2: ", np.shape(p2))

num_particles = len(p1)

# calculate the distance
dist = calc_phi_ij(p1, p2)
# print(dist)

# hungarian method
row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist)

l = np.amax(dist) * 1.5  # lambda
# l = 1  # lambda

print("lambda: ", l)

Q_ = Q(dist, l)
# np.savetxt('Q.txt', Q_, delimiter=',')
# print("Q:\n", Q_)
# Q_dic = Q_dict(Q_)
# print(np.amax(Q_))

token = 'DEV-3fd7a21d8cf1afa9655ac1d7e9cb809bc3d7f7dc'
solutions, energies, num_oc = QBSolve_quantum_solution(Q_, token=token)

solv_list = QBSolve_classical_solution(Q_)
sliced_solutions = solution_slicer(solv_list, num_particles)
max_outputs = find_argmax(sliced_solutions)

print(max_outputs)