"""
Script to found the bound particles of a satellite
galaxy.

author: Nicolás Garavito-Camargo

"""


import numpy as np
import schwimmbad
import coeff_parallel as cop
import parallel_pot 


def parallel_potential_batches(pos, S, T, rs, nmax, lmax, G, ncores,
                               npart_sample):
    """
    Function to compute potential in parallel. To avoid 
    large memory errors the particles are computed in batches.
    """
    
    Nparticles = len(pos[:,0])
    # compute n batches to split the particles
    nbatches = int(Nparticles/npart_sample)
    print('computing potential in parallel in {} batches of particles'.format(nbatches))
    pot_all = np.array([])
    for i in range(nbatches):
        if i < (nbatches-1):
            pos_batch = pos[i*npart_sample:(i+1)*npart_sample]
        if i == (nbatches-1):
            pos_batch = pos[i*npart_sample:]
        print("Done batch {}".format(i))
        halo_pot = parallel_pot.PBFEpot(pos_batch, S, T, rs, nmax, lmax, G=43007.1, M=1)
        pool2 = schwimmbad.choose_pool(mpi=False, processes=ncores)
        pot = halo_pot.main(pool2)
        pot_all = np.hstack((pot_all, pot))

    return pot_all

def compute_scf_pot(pos, rs, nmax, lmax, mass, ncores, npart_sample):
    """
    TODO: Parallelize this function here!
    """
    # Compute coefficients
    # Compute potential
    #_gadget = 43007.1

    pool = schwimmbad.choose_pool(mpi=False, processes=ncores)
    halo_coeff = cop.Coeff_parallel(pos, mass, rs, False, nmax, lmax)
    results = halo_coeff.main(pool)
    S = results[:,0]
    T = results[:,1]
     
    pot = parallel_potential_batches(pos, S, T, rs, nmax, lmax, 
                                     43007.1, ncores, npart_sample)
    
    print("Done computing parallel potential")
    return pot, S, T


def bound_particles(pot, pos, vel, ids):
    vmag_lmc = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)
    dist_lmc = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    T = vmag_lmc**2/2
    V = pot

    lmc_bound = np.where(T+V<=0)[0]
    lmc_unbound = np.where(T+V>0)[0]

    return pos[lmc_bound], vel[lmc_bound], ids[lmc_bound], pos[lmc_unbound], vel[lmc_unbound], ids[lmc_unbound]


def find_bound_particles(pos, vel, mass, ids, rs, nmax, lmax, ncores, npart_sample):
    N_init = len(pos)
    pot, S, T = compute_scf_pot(pos, rs, nmax, lmax, mass, ncores, npart_sample)
    pos_bound, vel_bound, ids_bound, pos_unbound, vel_unbound, ids_unbound = bound_particles(pot, pos, vel, ids)
    N_bound = len(pos_bound)

    print('Number of bound particles prior iteration {}'.format(N_bound))

    print('Initial number of particles:', N_init)
    i=0
    while (np.abs(N_init-N_bound) > (0.01*N_init)):
        pot, S, T = compute_scf_pot(pos_bound, rs, nmax, lmax, mass, ncores, npart_sample)
        pos_bound, vel_bound, ids_bound, p_unb, v_unb, ids_unb = bound_particles(pot, pos_bound, vel_bound, ids_bound)   
        N_init = N_bound
        N_bound = len(pos_bound)
        i+=1
        print(N_init, N_bound)
        print('Number of bound particles in iteration {}: {}'.format(i, N_bound))
        pos_unbound = np.vstack((pos_unbound, p_unb))
        vel_unbound = np.vstack((vel_unbound, v_unb))
        ids_unbound = np.hstack((ids_unbound, ids_unb))
    return pos_bound, N_bound, ids_bound, pos_unbound, vel_unbound, ids_unbound, S, T





