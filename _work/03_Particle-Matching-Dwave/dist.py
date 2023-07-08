import numpy as np

def calc_phi_ij(coords_n, coords_n_minus_1):
    num_particles = np.shape(coords_n)[0]
    print('Compute distance function of {} particles'.format(num_particles))
    phi_ij = np.zeros((num_particles, num_particles))
    for i in range(np.shape(coords_n)[1]): # loop over x,y,z
        phi_ij += np.square(np.subtract.outer(coords_n[:,i], coords_n_minus_1[:,i]))
    phi_ij = np.sqrt(phi_ij)
    #print(phi_ij)
    return(phi_ij)

def find_argmax(list_of_lists):
    return np.apply_along_axis(np.argmax, axis=1, arr=list_of_lists)