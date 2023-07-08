from dwave.cloud import Client

# Define your Q matrix (quadratic terms)
Q = {(0, 0): -1, (0, 1): 2, (1, 1): -1}

# Convert Q matrix to an Ising model
h = {}
J = {}
for (i, j), value in Q.items():
    if i == j:
        h[i] = value
    else:
        J[(i, j)] = value

# Create a QUBO object
QUBO = {}
for i in range(max(Q.keys())[0] + 1):
    for j in range(max(Q.keys())[1] + 1):
        QUBO[(i, j)] = h.get(i, 0) + J.get((i, j), 0)

# Set up the D-Wave client
client = Client.from_config(token='YOUR_TOKEN')

# Submit the QUBO to the quantum annealer
solver = client.get_solver()
response = solver.sample_qubo(QUBO, num_reads=100)

# Interpret the results
for sample, energy in response.all_solutions(['sample', 'energy']):
    print(sample, energy)
