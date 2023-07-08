from q import Q
import numpy as np

d = np.array([[2, 9, 10, 10, 1, 6], [8, 1, 10, 5, 8, 5], [5, 6, 2, 7, 8, 3], [1, 7, 8, 3, 10, 9],[7, 3, 5, 8, 1, 6],[6, 4, 7, 2, 9, 1],])
l = np.amax(d) * 1.5
Q_ = Q(d, l)
print(Q_)
print(np.max(Q_))