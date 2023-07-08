import numpy as np
import time as time
#positions_30s = np.loadtxt('coords_30p0s_03_10.txt')
#positions_32p5s = np.loadtxt('coords_32p5s_03_10.txt')
path = r'C:\Users\oahre\OneDrive\RWTH\Simulation_Science\HiWi\CWP_SS23\2023-06-pro-voeren-gruppen\Group 5 - Quantum Annealing Particle Matching in DEM Simulations\_work_OAH\02_Hungarian_algorithm'
#positions_30s = np.loadtxt(path+'\coords_30p0s_02_30.txt')
#positions_32p5s = np.loadtxt(path+'\coords_32p5s_02_30.txt')
#positions_30s = np.loadtxt('coords_30p0s_01_3160.txt')
#positions_32p5s = np.loadtxt('coords_32p5s_01_3160.txt')
#positions_30s = np.loadtxt('coords_30p0s_10_17332.txt')
#positions_32p5s = np.loadtxt('coords_32p5s_10_17332.txt')

file_list = []
file_list.append(path + '\coords_30p0s_01_3160.txt')
file_list.append(path + '\coords_32p5s_01_3160.txt')
file_list.append(path + '\coords_30p0s_02_30.txt')
file_list.append(path + '\coords_32p5s_02_30.txt')
file_list.append(path + '\coords_30p0s_03_10.txt')
file_list.append(path + '\coords_32p5s_03_10.txt')
file_list.append(path + '\coords_30p0s_04_100.txt')
file_list.append(path + '\coords_32p5s_04_100.txt')
file_list.append(path + '\coords_30p0s_04_6320.txt')
file_list.append(path + '\coords_32p5s_04_6320.txt')
file_list.append(path + '\coords_30p0s_05_300.txt')
file_list.append(path + '\coords_32p5s_05_300.txt')
file_list.append(path + '\coords_30p0s_06_1000.txt')
file_list.append(path + '\coords_32p5s_06_1000.txt')
file_list.append(path + '\coords_30p0s_07_12641.txt')
file_list.append(path + '\coords_32p5s_07_12641.txt')
file_list.append(path + '\coords_30p0s_10_17332.txt')
file_list.append(path + '\coords_32p5s_10_17332.txt')
file_list.append(path + '\coords_30p0s_13_36308.txt')
file_list.append(path + '\coords_32p5s_13_36308.txt')

#print(type(positions_30s))
#print(np.shape(positions_30s))
#print(positions_30s.dtype)
# compute distacnce of particles between two datasets


def calc_phi_ij(coords_n, coords_n_minus_1):
    num_particles = np.shape(coords_n)[0]
    print('Compute distance function of {} particles'.format(num_particles))
    phi_ij = np.zeros((num_particles, num_particles))
    for i in range(3): # loop over x,y,z
        phi_ij += np.square(np.subtract.outer(coords_n[:,i], coords_n_minus_1[:,i]))
    phi_ij = np.sqrt(phi_ij)
    #print(phi_ij)
    return(phi_ij)

for i in range(0, len(file_list), 2):
    positions_30s = np.loadtxt(file_list[i])
    positions_32p5s = np.loadtxt(file_list[i+1])

    start_time = time.time()
    distances = calc_phi_ij(positions_30s, positions_32p5s)
    displacements = positions_32p5s - positions_30s
    import scipy as scipy
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(distances)
    end_time = time.time()
    print(np.shape(positions_30s))
    error = np.sum(distances[row_ind, col_ind])
    #print('row indices: ', row_ind)
    is_row_unique = np.all(1 == np.unique(row_ind, return_counts=True)[1])
    print('row unique : ', is_row_unique)
    #print('col indices: ', col_ind)
    is_col_unique = np.all(1 == np.unique(col_ind, return_counts=True)[1])
    print('col unique : ', is_col_unique)
    print('cummulative distance: ', error)
    assignment_diff = positions_30s[col_ind] - positions_32p5s[row_ind]
    print('runtime [s]: ', end_time - start_time)
#import matplotlib.pyplot as plt
#%matplotlib widget
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
##ax.scatter(particles_coords[10,:,0], particles_coords[10,:,1], particles_coords[10,:,2])
##colors = np.zeros_like(positions_30s[:,0])
##colors[np.where(positions_30s[:,0]>0)]=1
##c=colors
#colors = np.linspace(0,1,np.shape(positions_30s)[0])
#ax.scatter(positions_30s[:,0], positions_30s[:,1], positions_30s[:,2], s=150, marker="x", c=colors)
#ax.scatter(positions_32p5s[:,0], positions_32p5s[:,1], positions_32p5s[:,2], s=150, c=colors)
#ax.quiver(positions_30s[:,0], positions_30s[:,1], positions_30s[:,2], displacements[:,0], displacements[:,1], displacements[:,2])
#ax.quiver(positions_32p5s[:,0], positions_32p5s[:,1], positions_32p5s[:,2], assignment_diff[:,0], assignment_diff[:,1], assignment_diff[:,2])
#ax.set_aspect('equal')