import dimod
import numpy as np
from dwave.system import LeapHybridSampler, EmbeddingComposite
# import dwave.inspector
import time


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
    # print("Q_DICT: ", Qdict)
    return Qdict

def QBSolve_classical_solution(Q):

    Q_d = Q_dict(Q)

    start = time.time()
    sim = dimod.SimulatedAnnealingSampler().sample_qubo(Q_d)
    end = time.time()
    print("Simulated Annealing time: ", end - start)

    energies = [list(sim.data(['energy']))[i].energy for i in range(len(sim))]
    min_energy_index = np.argmin(energies)

    decisions = [list(dict(list(sim.samples())[i]).values()) for i in range(len(sim))]
    return decisions[min_energy_index]


def QBSolve_quantum_solution(Q, token='DEV-3fd7a21d8cf1afa9655ac1d7e9cb809bc3d7f7dc'):

    Q_d = Q_dict(Q)

    sampler = LeapHybridSampler(solver='Advantage_system4.1', token=token)
    # sampler = EmbeddingComposite(solver={'category': 'hybrid'}, token=token)
    sim = sampler.sample_qubo(Q_d)

    # dwave.inspector.show(sim)

    energies = [list(sim.data(['energy']))[i].energy for i in range(len(sim))]
    min_energy_index = np.argmin(energies)

    decisions = [list(dict(list(sim.samples())[i]).values()) for i in range(len(sim))]
    return decisions[min_energy_index]


