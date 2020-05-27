"""
Script to found the bound particles of a satellite
galaxy using the BFE method.

author: Nicolás Garavito-Camargo

"""


import numpy as np
import schwimmbad
import coeff_parallel as cop
import parallel_pot 
from scipy import optimize



def enclosed_mass(pos, m):
    r = np.sqrt(np.sum(pos**2, axis=1))
    r_bins = np.arange(0, 30, 1)
    dr = (r_bins[1]-r_bins[0])/2.
    nbins = len(r_bins)
    rho = np.zeros(nbins-1)
    m_tot = np.zeros(nbins)
    m_cumulative = np.zeros(nbins)
    for i in range(nbins-1):
        index = np.where((r<r_bins[i+1]) & (r>=r_bins[i]))[0]
        m_shell = np.sum(m[index])
        v = 4/3. * np.pi * (r_bins[i+1]**3 - r_bins[i]**3)
        rho[i] = m_shell/v
        m_tot[i] = m_shell
        m_cumulative[i] = np.sum(m_tot[:i+1])
    return rho, r_bins[:-1] + dr, m_tot[:-1], m_cumulative[:-1]
                                                                                                


def hernquist_mass(r, a, M):
    return M*r**2. / (r+a)**2.


def find_scale_length(pos, mass, rs_max):
    rho, r, m, m_c = enclosed_mass(pos, mass)
    M = 4*np.sum(mass)
    rs_opt = optimize.curve_fit(hernquist_mass, r, m_c, bounds=(0, [rs_max, M]))
    print(rs_opt[0])
    return rs_opt[0][0]



def parallel_potential_batches(pos, S, T, rs, nmax, lmax, G, ncores,
                               npart_sample):
    """
    Function to compute potential in parallel. To avoid 
    large memory errors the particle's potential is computed in batches.

    Parameters:
    ----------
    pos
    S : cosine components
    T : sine components
    rs : halo scale length
    nmax : maximum of n number in expansion
    lmax : maximum l number in expansion
    G : value of gravitational constant
    ncores : number of cores used to compute the potential in parallel
    npart_sample : number of particles 

    """
    
    Nparticles = len(pos[:,0])
    # compute n batches to split the particles
    nbatches = int(Nparticles/npart_sample)
    print('computing potential in parallel in {} batches of particles'.format(nbatches))
    pot_all = np.array([])

    if nbatches < 1:
      pos_batch = pos
      halo_pot = parallel_pot.PBFEpot(pos_batch, S, T, rs, nmax, lmax, G=G, M=1)

      # choosing pool to compute the potential in parallel
      pool2 = schwimmbad.choose_pool(mpi=False, processes=ncores)
      # Compute potential in parallel
      pot_all = halo_pot.main(pool2)

    else :
        for i in range(nbatches):
            # making batches
            if i < (nbatches-1):
                pos_batch = pos[i*npart_sample:(i+1)*npart_sample]
            elif i == (nbatches-1):
                pos_batch = pos[i*npart_sample:]
            else :
                pos_batch = pos

            print("Done batch {}".format(i))
            halo_pot = parallel_pot.PBFEpot(pos_batch, S, T, rs, nmax, lmax, G=G, M=1)
            pool2 = schwimmbad.choose_pool(mpi=False, processes=ncores)
            pot = halo_pot.main(pool2)
            pot_all = np.hstack((pot_all, pot))

            # clean memory
            del(pot, pos_batch, pool2)
    # clean memory
    del(pos)

    assert (len(pot_all)==Nparticles), 'Hey some potentials are missing here'
    return pot_all

def compute_scf_pot(pos, rs, nmax, lmax, mass, ncores, npart_sample):
    """
    
    """
    # Compute coefficients
    # Compute potential
    G_gadget = 43007.1

    # Compute coefficients of bound particles
    pool = schwimmbad.choose_pool(mpi=False, processes=ncores)
    assert len(pos)==len(mass), 'position array and mass array length do not match'
    rs_opt = find_scale_length(pos, mass, rs_max=rs)
    print(rs_opt, rs)
    rs_opt = rs
    # TODO: make rs_max to be an opt parameter
    print("rs", rs_opt)
    halo_coeff = cop.Coeff_parallel(pos, mass, rs_opt, False, nmax, lmax)
    results = halo_coeff.main(pool)
    S = results[:,0]
    T = results[:,1]

    del(results, halo_coeff, pool)

    # Compute potential
    pot = parallel_potential_batches(pos, S, T, rs_opt, nmax, lmax, 
                                     G_gadget, ncores, npart_sample)
    del(S,T,pos)
    print("Done computing parallel potential")
    return pot, rs_opt


def bound_particles(pot, pos, vel, ids, rcut=0):
    """
    Find bound particles
    TODO: Make rcut input paramter
    """
    vmag_lmc = np.sqrt(np.sum(vel**2, axis=1))
    T = vmag_lmc**2/2
    V = pot
    r = np.sqrt(np.sum(pos**2, axis=1))
    lmc_bound = np.where((T+V<=0) | (r<=rcut))[0]
    lmc_unbound = np.where((T+V>0) & (r>rcut))[0]
    del(T, V, vmag_lmc)

    return pos[lmc_bound], vel[lmc_bound], ids[lmc_bound], pos[lmc_unbound], vel[lmc_unbound], ids[lmc_unbound]


def find_bound_particles(pos, vel, mass, ids, rs, nmax, lmax, ncores, npart_sample):
    """
    Iterating to find bound particles
    mass : particle mass array
    """
    N_init = len(pos)
    path_temp = '/home/u9/jngaravitoc/codes/BFE_run_scripts/MWLMC6/'
    pot, rs_opt = compute_scf_pot(pos, rs, nmax, lmax, mass, ncores, npart_sample)
    pos_bound, vel_bound, ids_bound, pos_unbound, vel_unbound, ids_unbound = bound_particles(pot, pos, vel, ids)
    N_bound = len(ids_bound)
    np.savetxt(path_temp +'bound_particles_iteration_init_{:0>2d}.txt'.format(0), pos_bound, header=str(rs_opt))
    mp = mass[0]
    del(pos)
    del(vel)
    del(ids)
    del(mass)

    print('Initial number of particles:', N_init)
    print('Number of bound particles prior iteration {}'.format(N_bound))
    #print("bound mass", N_bound*mass[0])
    i=1
    
    while (np.abs(N_init-N_bound) > (0.01*N_init)):
        pot, rs_opt = compute_scf_pot(pos_bound, rs, nmax, lmax, np.ones(N_bound)*mp, ncores, npart_sample)
        pos_bound, vel_bound, ids_bound, p_unb, v_unb, ids_unb = bound_particles(pot, pos_bound, vel_bound, ids_bound)   
        N_init = N_bound
        N_bound = len(ids_bound)
        print("bound mass", N_bound*mp)
        i+=1
        print(N_init, N_bound)
        #print('Number of bound particles in iteration {}: {}'.format(i, N_bound))
        pos_unbound = np.vstack((pos_unbound, p_unb))
        vel_unbound = np.vstack((vel_unbound, v_unb))
        ids_unbound = np.hstack((ids_unbound, ids_unb))
        
        np.savetxt(path_temp +
            'bound_particles_iteration_init_{:0>2d}.txt'.format(i-1), pos_bound, header=str(rs_opt))
        del(p_unb)
        del(v_unb)
        del(ids_unb)
   
    return pos_bound, vel_bound, ids_bound, pos_unbound, vel_unbound, ids_unbound, rs_opt#N_bound
    #return pos, vel, ids, np.zeros(len(ids)), np.zeros(len(ids)), mass, pot#N_bound





