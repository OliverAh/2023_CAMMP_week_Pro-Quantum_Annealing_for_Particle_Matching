from collections import Counter
from qbsolv2 import QBSolve_classical_solution
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.cloud import Client
from qbsolv import solution_slicer
from dist import find_argmax
import numpy as np
import dwave_qbsolv as QBSolv
import time
import matplotlib.pyplot as plt

def sanity_check(sol):
    # check if the solutions are sane
    if len(sol) == len(set(sol)):
        return True
    else:
        return False

# sort the indices of the values
def index_sort(values):
    return [i for i, _ in sorted(enumerate(values), key=lambda x: x[1], reverse=True)]

def counter(list):
    return dict(Counter(map(tuple, list)))

def add_inverse_and_zero(matrix, number, p):
    # Flatten the matrix into a 1D list
    flattened_matrix = np.array(matrix).flatten()

    # Add the number and inverse all the elements
    inverted_matrix = 1 / (flattened_matrix + number)

    # Calculate the number of elements to set to zero
    num_elements_to_zero = int(p * len(inverted_matrix) / 100)

    # Set a percentage of the smallest values to zero
    sorted_indices = np.argsort(inverted_matrix)
    zero_indices = sorted_indices[:num_elements_to_zero]
    inverted_matrix[zero_indices] = 0

    # Reshape the flattened matrix back into the original shape
    updated_matrix = inverted_matrix.reshape(matrix.shape)

    return updated_matrix



def Q_dict(Q):
    """This function changes the Q from matrix form to Dict form usable by QBSolv"""
    keys = []
    QDist_list = []

    for i in range(len(Q[0])):
        for j in range(len(Q[0])):
            if Q[i][j] != 0:
                keys.append((i, j))
                QDist_list.append(Q[i][j])

    Qdict = {keys[i]: QDist_list[i] for i in range(len(keys))}
    return Qdict

def classical(Q):

    """This function use classical QBSolve to get solution dictionary"""

    Qdict = Q_dict(Q)

    s = time.time()
    response = QBSolv.QBSolv().sample_qubo(Qdict, max_iter=1000)
    e = time.time()
    print("Time taken by Classical QBSolv: ", e - s)

    all_solutions = []

    # Iterate over the samples in the response
    for sample, energy, num_occurrences in response.data(['sample', 'energy', 'num_occurrences']):

        # getting solution from sample by converting the values of the dictionary to a list
        sample = list(sample.values())

        # slicing the list of solutions into a list of lists
        num_particles = int(np.sqrt(len(Q)))
        sliced_solutions = solution_slicer(sample, num_particles)

        # finding the argmax of the sliced solutions
        max_outputs = find_argmax(sliced_solutions)

        # Each sample is a dictionary where the keys are the variable labels and the values are the variable values
        solution = {
            'solution': max_outputs,
            'energy': energy,
            'num_occurrences': num_occurrences,
            'optimal_value': np.dot(sample, np.dot(Q, sample))
        }
        all_solutions.append(solution)

    return all_solutions

def quantum(Q, token="DEV-3fd7a21d8cf1afa9655ac1d7e9cb809bc3d7f7dc", solver_name="Advantage_system6.2", num_reads=1000):
    # Create a D-Wave sampler and specify your token and solver
    sampler = DWaveSampler(token=token, solver=solver_name)

    # Create an embedding composite. This helps map the problem to the hardware topology.
    embedded_sampler = EmbeddingComposite(sampler)

    # Submit the QUBO to the D-Wave system and specify the number of samples
    response = embedded_sampler.sample_qubo(Q, num_reads=num_reads)

    # Get the QPU processing time in microseconds
    qpu_time = response.info['timing']['qpu_access_time']

    print(f'QPU processing time: {qpu_time / 1e6} seconds')

    # Create a list to hold all solutions
    all_solutions = []

    # Iterate over the samples in the response
    for sample, energy, num_occurrences in response.data(['sample', 'energy', 'num_occurrences']):

        # getting solution from sample by converting the values of the dictionary to a list
        sample = list(sample.values())

        # slicing the list of solutions into a list of lists
        num_particles = int(np.sqrt(len(Q)))
        sliced_solutions = solution_slicer(sample, num_particles)

        # finding the argmax of the sliced solutions
        max_outputs = find_argmax(sliced_solutions)

        # Each sample is a dictionary where the keys are the variable labels and the values are the variable values
        solution = {
            'solution': max_outputs,
            'energy': energy,
            'num_occurrences': num_occurrences,
            'optimal_value': np.dot(sample, np.dot(Q, sample))
        }
        all_solutions.append(solution)

    # Return the list of all solutions
    return all_solutions

def plot_hist(all_solutions, correct_solution, title="Histogram of Energies"):
    # Sort all solutions by energy
    all_solutions.sort(key=lambda d: d["energy"])

    # Mapping energy to positive range (i.e., subtract minimum energy from all)
    min_energy = min(d["energy"] for d in all_solutions)
    for d in all_solutions:
        d["energy"] -= min_energy

    # Create the bars one by one, checking for correct solutions
    for i, d in enumerate(all_solutions):
        color = 'green' if np.array_equal(d["solution"], correct_solution) else 'red'
        plt.bar(i, d["num_occurrences"], color=color, width=0.2, zorder=3)

    # Set labels and title
    plt.xlabel("Solution Index (Sorted by Energy)")
    plt.ylabel("Number of Occurrences")
    plt.title(title)

    # Show the plot
    plt.show()

def correct_solution_counter(all_solutions, correct_solution):

    # count total number of solutions
    total_num_sols = np.sum([solution["num_occurrences"] for solution in all_solutions])

    # Initialize the number of correct solutions
    num_correct_solutions = 0

    # Iterate over all solutions
    for solution in all_solutions:
        # Check if the solution is correct
        if np.array_equal(solution["solution"], correct_solution):
            # Increment the number of correct solutions
            num_correct_solutions += solution["num_occurrences"]


    return num_correct_solutions/total_num_sols


def solvers_list(token="DEV-3fd7a21d8cf1afa9655ac1d7e9cb809bc3d7f7dc"):
    # Create a D-Wave client
    with Client(token=token) as client:
        # Get a list of solvers
        solvers = client.get_solvers()

        # Print out their names
        for solver in solvers:
            print(solver.id)