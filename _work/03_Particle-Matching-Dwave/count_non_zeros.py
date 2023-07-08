import numpy as np

# Create a 2D array
arr = np.loadtxt('Q.txt', dtype=int, delimiter=',')
print(arr)

# Count non-zero elements
count = np.count_nonzero(arr)

print("Number of non-zero elements:", count)
