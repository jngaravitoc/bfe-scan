"""
Script to found the bound particles of a satellite
galaxy.

author: Nicolás Garavito-Camargo

usage: python LMC_bounded.py filename nmax lmax rs

"""


import numpy as np
from gala.potential.scf._computecoeff import STnlm_discrete, STnlm_var_discrete
import gala.potential as gp

def reading_particles(snap_name):
    lmc_particles = np.loadtxt(snap_name)
    pos = lmc_particles[:,0:3]
    vel = lmc_particles[:,3:6]
    ids = lmc_particles[:,6]
    mass = lmc_particles[:,7]
    Mtot = np.sum(mass)
    #rand = np.random.randint(0, len(mass), 100000)
    #mass_rand_part = (Mtot/1E5)*np.ones(100000)
    #print('Total mass of the halo is:', np.sum(mass))
    #print('Total mass of the sampled halo is:', np.sum(mass_rand_part))
    #return pos[rand], vel[rand], mass_rand_part, ids[rand]
    return pos, vel, mass, ids


def compute_scf_pot(pos, rs, nmax, lmax, mass):
    """
    TODO: Parallelize this function here!
    """
    G_gadget = 43007.1
    S, T = STnlm_discrete(np.ascontiguousarray(pos).astype(float),
                                        mass, nmax, lmax, rs)

    LMC_potential = gp.potential(S, T, M=1, r_s=rs, G=G_gadget)
    potential = LMC_potential(np.ascontiguousarray(pos).astype(float))
    return potential


def bound_particles(pot, pos, vel, ids):
    vmag_lmc = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)
    dist_lmc = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    T = vmag_lmc**2/2
    V = pot

    lmc_bound = np.where(T+V<=0)[0]
    lmc_unbound = np.where(T+V>0)[0]

    return pos[lmc_bound], vel[lmc_bound], ids[lmc_bound], pos[lmc_unbound], vel[lmc_unbound], ids[lmc_unbound]


def find_bound_particles(pos, vel, mass, ids, rs, nmax, lmax):
    N_init = len(pos)
    pot = compute_scf_pot(pos, rs, nmax, lmax, mass)
    pos_bound, vel_bound, ids_bound, pos_unbound, vel_unbound, ids_unbound = bound_particles(pot, pos, vel, ids)
    N_bound = len(pos_bound)


    print('Initial number of particles:', N_init)
    i=0
    while (np.abs(N_init-N_bound) > (0.01*N_init)):
        pot = compute_scf_pot(pos_bound, rs, nmax, lmax, mass)
        pos_bound, vel_bound, ids_bound, p_unb, v_unb, ids_unb = bound_particles(pot, pos_bound, vel_bound, ids_bound)   
        N_init = N_bound
        N_bound = len(pos_bound)
        i+=1
        print(N_init, N_bound)
        print('Number of bound particles in iteration {}: {}'.format(i, N_bound))
        pos_unbound = np.vstack((pos_unbound, p_unb))
        vel_unbound = np.vstack((vel_unbound, v_unb))
        ids_unbound = np.hstack((ids_unbound, ids_unb))
    return pos_bound, vel_bound, N_bound, ids_bound, pos_unbound, vel_unbound, ids_unbound


if __name__ == "__main__":

   
    snapname = sys.argv[1]
    out_name = sys.argv[2]
    nmax = int(sys.argv[3])
    lmax = int(sys.argv[4])
    rs = float(sys.argv[5])

    
    pos, vel, mass, ids = reading_particles(snapname)
    print('Snapshot loaded')
    print(pos[0])
    print(vel[0])
    print(mass[0])
    armadillo = find_bound_particles(pos, vel, mass, ids, rs, nmax, lmax)
    print('Bound particles computed')

    pos_bound = armadillo[0]
    vel_bound = armadillo[1]
    N_bound =  armadillo[2]
    ids_bound =  armadillo[3]
    pos_unbound = armadillo[4]
    vel_unbound = armadillo[5]
    ids_unbound = armadillo[6]
    
    lmc_bound = np.array([pos_bound[:,0], pos_bound[:,1], pos_bound[:,2], 
                          vel_bound[:,0], vel_bound[:,1], vel_bound[:,2],
                          ids_bound]).T
    
    lmc_unbound = np.array([pos_unbound[:,0], pos_unbound[:,1], pos_unbound[:,2], 
                            vel_unbound[:,0], vel_unbound[:,1], vel_unbound[:,2],
                            ids_unbound]).T
    
    np.savetxt(out_name, lmc_bound)
    print('Done writing snapshot with satellite bounded particles')
    np.savetxt("unbound"+out_name, lmc_unbound)
    print('Done writing snapshot with satellite unbounded particles')




