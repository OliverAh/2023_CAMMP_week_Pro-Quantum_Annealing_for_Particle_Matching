import numpy as np
from qbsolv import QBSolve_classical_solution, solution_slicer, QBSolve_quantum_solution
from dist import calc_phi_ij, find_argmax
import scipy
from q import Q

# token = ''

# load the coordinate files
# p1 = np.loadtxt('coords_30p0s_02_30.txt')
# p2 = np.loadtxt('coords_32p5s_02_30.txt')
p1 = np.loadtxt('coords_30p0s_03_10.txt')
p2 = np.loadtxt('coords_32p5s_03_10.txt')

num_particles = len(p1)

# calculate the distance
dist = calc_phi_ij(p1, p2)

# hungarian method
row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist)
l = np.max(dist) * 1.1  # lambda

print("lambda: ", l)

Q = Q(dist, l)

# Quantum Annealing
# qc_output = QBSolve_quantum_solution(Q, token)
# qc_sliced_sols = solution_slicer(qc_output, num_particles)
# qc_max_outputs = find_argmax(qc_sliced_sols)

# Simulated Annealing
sim_output = QBSolve_classical_solution(Q)
sim_sliced_sols = solution_slicer(sim_output, num_particles)
sim_max_outputs = find_argmax(sim_sliced_sols)
# print(max_outputs)

print(f'N : SIM ,Hung')
for i,o in enumerate(sim_max_outputs):
    print(f'{i} : {o} ,{col_ind[i]}')

# Use the following code when using the quantum annealer
# print(f'N : QC, SIM ,Hung')
# for i,o in enumerate(sim_max_outputs):
#     print(f'{i} : {o}, {sim_max_outputs[i]} ,{col_ind[i]}')
# print("QC solv: ", np.sum(dist[row_ind, qc_max_outputs]))

print("hung dist: ", np.sum(dist[row_ind, col_ind]))
print("Sim solv: ", np.sum(dist[row_ind, sim_max_outputs]))