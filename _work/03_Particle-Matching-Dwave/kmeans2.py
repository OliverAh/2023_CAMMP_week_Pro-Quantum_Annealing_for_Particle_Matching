from sklearn.cluster import KMeans
from q import Q
import numpy as np
from dist import calc_phi_ij

def q_cluster(dot_coords, cross_coords, n_clusters):
    # Concatenate the dot and cross coordinates
    coords = np.concatenate([dot_coords, cross_coords], axis=0)

    # Cluster the coordinates using k-means
    kmeans = KMeans(n_clusters=n_clusters).fit(coords)

    # Determine the cluster of each dot and cross
    dot_clusters = kmeans.predict(dot_coords)
    cross_clusters = kmeans.predict(cross_coords)

    # Initialize list to store Q matrices
    Q_matrices = []
    dot_indices_list = []
    cross_indices_list = []

    # For each cluster, solve the sub-problem
    for i in range(n_clusters):
        # Get the dots and crosses in this cluster
        dot_indices = np.where(dot_clusters == i)[0]
        cross_indices = np.where(cross_clusters == i)[0]

        dot_indices_list.append(dot_indices)
        cross_indices_list.append(cross_indices)

        # Calculate distance matrix for this sub-problem
        d_sub = calc_phi_ij(dot_coords[dot_indices], cross_coords[cross_indices])


        # Construct and store the Q matrix for this sub-problem
        Q_sub = Q(d_sub)
        Q_matrices.append(Q_sub)

    # Return the list of Q matrices
    return Q_matrices, dot_indices_list, cross_indices_list


p1 = np.array([[1.5,1],[1.5,2],[1.5,3],[1.5,4]])
p2 = np.array([[1,1],[1,2],[1,3],[1,4]])

c = q_cluster(p1,p2,2)
print(c)