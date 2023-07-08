from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.cloud import Client


# Create a D-Wave sampler instance with the desired solver
sampler = DWaveSampler()

# Connect to the D-Wave cloud client
client = Client.from_config()

# Get the available solvers
solvers = client.get_solvers()
for solver in solvers:
    print(solver)
