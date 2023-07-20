import numpy as np
import scipy as scipy
import time as time
#positions_30s = np.loadtxt('coords_30p0s_03_10.txt')
#positions_32p5s = np.loadtxt('coords_32p5s_03_10.txt')
path = r'C:\Users\adam-1aeqn8vhvpjnv4u\OneDrive - Students RWTH Aachen University\RWTH\Simulation_Sciences\HiWi\02_Hungarian_algorithm'
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

file_list.append(path + '\coords_30p0s_07_72101.txt')
file_list.append(path + '\coords_32p5s_07_72101.txt')
file_list.append(path + '\coords_30p0s_07_142691.txt')
file_list.append(path + '\coords_32p5s_07_142691.txt')
file_list.append(path + '\coords_30p0s_07_284982.txt')
file_list.append(path + '\coords_32p5s_07_284982.txt')
file_list.append(path + '\coords_30p0s_07_569090.txt')
file_list.append(path + '\coords_32p5s_07_569090.txt')
file_list.append(path + '\coords_12p5s_07_1239132.txt')
file_list.append(path + '\coords_15p0s_07_1239132.txt')

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
    print('Finished computing distances')
    #print(phi_ij)
    return(phi_ij)

dict_timing = {}

for i in range(0, len(file_list), 2):
    positions_30s = np.loadtxt(file_list[i])
    positions_32p5s = np.loadtxt(file_list[i+1])

    print(' ')
    num_particles = np.shape(positions_30s)[0]
    print(num_particles)
    
    start_time_dist = time.time()
    distances = calc_phi_ij(positions_32p5s, positions_30s)
    end_time_dist = time.time()
    
    displacements = positions_32p5s - positions_30s
    
    start_time_hung = time.time()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(distances)
    end_time_hung = time.time()
    
    error = np.sum(distances[row_ind, col_ind])
    #print('row indices: ', row_ind)
    is_row_unique = np.all(1 == np.unique(row_ind, return_counts=True)[1])
    print('row unique : ', is_row_unique)
    #print('col indices: ', col_ind)
    is_col_unique = np.all(1 == np.unique(col_ind, return_counts=True)[1])
    print('col unique : ', is_col_unique)
    print('cummulative distance: ', error)
    assignment_diff = positions_30s[col_ind] - positions_32p5s[row_ind]
    runtime_dist = end_time_dist - start_time_dist
    runtime_hung = end_time_hung - start_time_hung
    runtime_all = runtime_dist + runtime_hung
    print('runtime dist[s]: ', runtime_dist)
    print('runtime hung[s]: ', runtime_hung)
    print('runtime all[s]: ', runtime_all)
    dict_timing[str(num_particles)] = [runtime_dist, runtime_hung, runtime_all]
    
    
    
print(dict_timing)
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