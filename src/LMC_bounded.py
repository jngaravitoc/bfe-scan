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
    large memory errors the particle's potential is computed in batches.
    """
    
    Nparticles = len(pos[:,0])
    # compute n batches to split the particles
    nbatches = int(Nparticles/npart_sample)
    print('computing potential in parallel in {} batches of particles'.format(nbatches))
    pot_all = np.array([])
    if nbatches < 1:
      pos_batch = pos
      halo_pot = parallel_pot.PBFEpot(pos_batch, S, T, rs, nmax, lmax, G=43007.1, M=1)
      pool2 = schwimmbad.choose_pool(mpi=False, processes=ncores)
      pot_all = halo_pot.main(pool2)

    else :
        for i in range(nbatches):
            if i < (nbatches-1):
                pos_batch = pos[i*npart_sample:(i+1)*npart_sample]
            elif i == (nbatches-1):
                pos_batch = pos[i*npart_sample:]
            else :
                pos_batch = pos
            print("Done batch {}".format(i))
            halo_pot = parallel_pot.PBFEpot(pos_batch, S, T, rs, nmax, lmax, G=43007.1, M=1)
            pool2 = schwimmbad.choose_pool(mpi=False, processes=ncores)
            pot = halo_pot.main(pool2)
            pot_all = np.hstack((pot_all, pot))
            del(pot)
            del(pos_batch)

    assert (len(pot_all)==Nparticles), 'Hey some potentials are missing here'
    return pot_all

def compute_scf_pot(pos, rs, nmax, lmax, mass, ncores, npart_sample):
    """
    TODO: Parallelize this function here!
    """
    # Compute coefficients
    # Compute potential
    #_gadget = 43007.1

    # Compute coefficients of bound particles
    pool = schwimmbad.choose_pool(mpi=False, processes=ncores)
    assert len(pos)==len(mass), 'positions arrays and massed do not match'
    halo_coeff = cop.Coeff_parallel(pos, mass, rs, False, nmax, lmax)
    results = halo_coeff.main(pool)
    S = results[:,0]
    T = results[:,1]
     
    # Compute potential
    pot = parallel_potential_batches(pos, S, T, rs, nmax, lmax, 
                                     43007.1, ncores, npart_sample)
    
    print("Done computing parallel potential")
    return pot


def bound_particles(pot, pos, vel, ids):
    """
    Find bound particles
    """
    vmag_lmc = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)
    T = vmag_lmc**2/2
    V = pot

    lmc_bound = np.where(T+V<=0)[0]
    lmc_unbound = np.where(T+V>0)[0]
    del(T)
    del(V)
    del(vmag_lmc)

    return pos[lmc_bound], vel[lmc_bound], ids[lmc_bound], pos[lmc_unbound], vel[lmc_unbound], ids[lmc_unbound]


def find_bound_particles(pos, vel, mass, ids, rs, nmax, lmax, ncores, npart_sample):
    """
    Iterating to find bound particles
    mass : particle mass array
    """
    N_init = len(pos)
    pot = compute_scf_pot(pos, rs, nmax, lmax, mass, ncores, npart_sample)
    #pos_bound, vel_bound, ids_bound, pos_unbound, vel_unbound, ids_unbound = bound_particles(pot, pos, vel, ids)
    #N_bound = len(ids_bound)
    #del(pos)
    #del(vel)
    #del(ids)
    print('Initial number of particles:', N_init)
    #print('Number of bound particles prior iteration {}'.format(N_bound))
    #print("bound mass", N_bound*mass[0])
    i=0
    """
    while (np.abs(N_init-N_bound) > (0.01*N_init)):
        pot = compute_scf_pot(pos_bound, rs, nmax, lmax, np.ones(N_bound)*mass[0], ncores, npart_sample)
        pos_bound, vel_bound, ids_bound, p_unb, v_unb, ids_unb = bound_particles(pot, pos_bound, vel_bound, ids_bound)   
        N_init = N_bound
        N_bound = len(ids_bound)
        print("bound mass", N_bound*mass[0])
        i+=1
        print(N_init, N_bound)
        print('Number of bound particles in iteration {}: {}'.format(i, N_bound))
        pos_unbound = np.vstack((pos_unbound, p_unb))
        vel_unbound = np.vstack((vel_unbound, v_unb))
        ids_unbound = np.hstack((ids_unbound, ids_unb))
        del(p_unb)
        del(v_unb)
        del(ids_unb)
    """
    #return pos_bound, vel_bound, ids_bound, pos_unbound, vel_unbound, ids_unbound, pot#N_bound
    return pos, vel, ids, np.zeros(len(ids)), np.zeros(len(ids)), mass, pot#N_bound





