import dimod
import numpy as np
from dwave.system import LeapHybridSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite


def bqm_run(Q):
    token = 'DEV-3fd7a21d8cf1afa9655ac1d7e9cb809bc3d7f7dc'
    # sampler = LeapHybridSampler(solver={'category': 'hybrid'}, token='DEV-3fd7a21d8cf1afa9655ac1d7e9cb809bc3d7f7dc')
    sampler = EmbeddingComposite(DWaveSampler(token=token))
    # Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2, (0,2): 4}
    # bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset = 0.0)
    bqm = dimod.BQM({}, Q, 0, dimod.Vartype.BINARY)
    sampleset = sampler.sample(bqm, num_reads=1000)
    return sampleset

# sampler = dimod.ExactSolver().sample_qubo(Q)
# print(sampler.sample_qubo(Q))

# sim = dimod.SimulatedAnnealingSampler().sample_qubo(Q)
#
# energys = [list(sim.data(['energy']))[i].energy for i in range(len(sim))]
# min_energy_index = np.argmin(energys)
# print(energys)
# print(min_energy_index)
#
# decisions = [list(dict(list(sim.samples())[i]).values()) for i in range(len(sim))]
# print(decisions[min_energy_index])

