#%%
import numpy as np
from qbsolv import solution_slicer, QBSolve_quantum_solution, Q_dict
# from qbsolv2 import QBSolve_classical_solution, QBSolve_quantum_solution
from dist import calc_phi_ij, find_argmax
from access_test import bqm_run
import scipy
from q import Q
from misc import sanity_check, index_sort, counter

# token = ''

# load the coordinate files
# p1 = np.loadtxt('coords_30p0s_02_30.txt')
# p2 = np.loadtxt('coords_32p5s_02_30.txt')
# p1 = np.loadtxt('coords_30p0s_03_10.txt')
# p2 = np.loadtxt('coords_32p5s_03_10.txt')

p1 = np.array([[1.5,1],[1.5,2],[1.5,3],[3,1],[3,2],[3,3]])
p2 = np.array([[1,1],[1,2],[1,3],[2.5,1],[2.5,2],[2.5,3]])

# p1 = np.array([[1.5,1],[1.5,2],[1.5,3]])
# p2 = np.array([[1,1],[1,2],[1,3]])

print("shape of p1: ", np.shape(p1))
print("shape of p2: ", np.shape(p2))

num_particles = len(p1)

# calculate the distance
dist = calc_phi_ij(p1, p2)
print(dist)


# dist = np.array([[2, 9, 10, 10, 1, 6], [8, 1, 10, 5, 8, 5], [5, 6, 2, 7, 8, 3], [1, 7, 8, 3, 10, 9],[7, 3, 5, 8, 1, 6],[6, 4, 7, 2, 9, 1],])
# dist = np.array([[2, 9, 10, 10, 1, 6], [8, 1, 10, 5, 8, 5], [5, 6, 2, 7, 8, 3], [1, 7, 8, 3, 10, 9],[7, 3, 5, 8, 1, 6],[6, 4, 7, 2, 9, 1],])


# num_particles = len(dist)

# hungarian method
row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist)


l = np.amax(dist) * 1.5  # lambda
# l = 1  # lambda

print("lambda: ", l)

Q_ = Q(dist, l)
np.savetxt('Q.txt', Q_, delimiter=',')
# print("Q:\n", Q_)
Q_dic = Q_dict(Q_)
# print(np.amax(Q_))

#%%
token = 'DEV-3fd7a21d8cf1afa9655ac1d7e9cb809bc3d7f7dc'
solutions, energies, num_oc = QBSolve_quantum_solution(Q_, token=token)

# Sorting the indicies by num_oc
num_oc_indicies = index_sort(num_oc)
print("num_oc_indicies: ", num_oc_indicies)

#%%
print(len(solutions))

# #%%
# bqm_sol = bqm_run(Q_dic)
#
# print(bqm_sol)
#
# #%%
# print(bqm_sol.data)
#
# #%%
# s = [list(dict(list(bqm_sol.samples())[i]).values()) for i in range(len(bqm_sol))]
# print(s)
#
# #%%
# energies = [list(bqm_sol.data(['energy']))[i].energy for i in range(len(bqm_sol))]
# min_energy_index = np.argmin(energies)
# min_s = s[min_energy_index]
#
# #%%
# sim_sliced_sols = solution_slicer(min_s, num_particles)
# sim_max_outputs = find_argmax(sim_sliced_sols)
# print(sim_max_outputs)

#%%

counter_dict = counter(solutions)

print("count dict: ", counter_dict)


# checking if the energy is the same for all the same solutions
# for sol in counter_dict.keys():
#     indices = np.where([np.array_equal(x, sol) for x in solutions])[0]
#     print("Solution: ", sol)
#     print("Occurrences: ", counter_dict[sol])
#     print("num of same energies: ", len(indices))
#     eners = [energies[i] for i in indices]
#     print("energies: ", eners)


solutions = [key for key, _ in counter_dict.items()]
num_oc = [value for _, value in counter_dict.items()]
num_oc_indicies = index_sort(counter_dict.values())



#%%
hung_dist = np.sum(dist[row_ind, col_ind])
print("Hungarian distance: ", hung_dist)

sane_solutions = []
sane_errors = []

# loop over all the solutions in the order of most num_oc to least
for i in range(len(num_oc_indicies)):
    sol = list(solutions[num_oc_indicies[i]])

    # Quantum Annealing
    # qc_output = QBSolve_quantum_solution(Q)
    qc_sliced_sols = solution_slicer(sol, num_particles)
    qc_max_outputs = find_argmax(qc_sliced_sols)

    # check sanity
    sane = sanity_check(qc_max_outputs)
    if sane:
        sane_solutions.append(qc_max_outputs)
        error_dist = np.sum(dist[row_ind, qc_max_outputs])
        sane_errors.append(error_dist)


    # print("QC outputs: ",qc_max_outputs)
    #
    # # Simulated Annealing
    # # sim_output = QBSolve_classical_solution(Q)
    # # sim_sliced_sols = solution_slicer(sim_output, num_particles)
    # # sim_max_outputs = find_argmax(sim_sliced_sols)
    # # print(max_outputs)
    #

    #
    # # Use the following code when using the quantum annealer
    # # print(f'N : QC, SIM ,Hung')
    # # for i,o in enumerate(sim_max_outputs):
    # #     print(f'{i} : {o}, {sim_max_outputs[i]} ,{col_ind[i]}')
    # # print("QC solv: ", np.sum(dist[row_ind, qc_max_outputs]))

    qc_dist = np.sum(dist[row_ind, qc_max_outputs])

    # print("Sim solv: ", np.sum(dist[row_ind, sim_max_outputs]))
    print("QC solution: ", qc_max_outputs)
    print("num_oc: ", num_oc[num_oc_indicies[i]])
    print("QC dist: ", qc_dist)
    # print("Energy: ", energies[num_oc_indicies[i]])
    print("Sanity: ", sane)
    print()

print("number of sane solutions: ", len(sane_solutions))

# find the minimum error
min_error_index = np.argmin(sane_errors)
print("min sane error: ", sane_errors[min_error_index])
min_sane_solution = sane_solutions[min_error_index]
print("min sane solution: ", min_sane_solution)




print(f'N : SIM ,Hung')
for i,o in enumerate(min_sane_solution):
    print(f'{i} : {o} ,{col_ind[i]}')