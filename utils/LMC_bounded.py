"""
Script to found the bound particles of a satellite
galaxy.

author: Nicolás Garavito-Camargo

usage: python LMC_bounded.py filename nmax lmax rs

"""


import numpy as np
from gala.potential.scf import compute_coeffs_discrete
from gala.potential.scf._bfe import potential as gala_scf_potential
import sys

def reading_particles(snap_name):
    lmc_particles = np.loadtxt(snap_name)
    pos = lmc_particles[:,0:3]
    vel = lmc_particles[:,3:6]
    mass = lmc_particles[:,6]
    print('Total mass of the halo is:', np.sum(mass))
    return pos, vel, mass


def compute_scf_pot(pos, rs, nmax, lmax, mass):
    G_gadget = 43007.1
    S, T = compute_coeffs_discrete(np.ascontiguousarray(pos).astype(float),
                                        mass, nmax, lmax, rs)

    LMC_potential = gala_scf_potential(np.ascontiguousarray(pos), S, T, M=1, r_s=rs, G=G_gadget)
    return LMC_potential


def compute_scf_pot_parallel(pos, rs, nmax, lmax, mass):
    """

    """
    G_gadget = 43007.1
    print('Computing coefficients')
    S, T = compute_coeffs_discrete(np.ascontiguousarray(pos).astype(float),
                                        mass, nmax, lmax, rs)
    print('Computing potential')
    LMC_potential = gala_scf_potential(np.ascontiguousarray(pos), S, T, M=1, r_s=rs, G=G_gadget)
    return LMC_potential


def bound_particles(pot, pos, vel):
    vmag_lmc = np.sqrt(vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2)
    dist_lmc = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    T = vmag_lmc**2/2
    V = pot

    lmc_bound = np.where(T+V<=0)[0]

    return pos[lmc_bound], vel[lmc_bound]


def find_bound_particles(pos, vel, mass, rs, nmax, lmax):
    N_init = len(pos)
    pot = compute_scf_pot(pos, rs, nmax, lmax, mass)
    pos_bound, vel_bound = bound_particles(pot, pos, vel)
    N_bound = len(pos_bound)

    print('Initial number of particles:', N_init)
    i=0
    while (np.abs(N_init-N_bound) > (0.01*N_init)):
        pot = compute_scf_pot(pos_bound, rs, nmax, lmax, mass)
        pos_bound, vel_bound = bound_particles(pot, pos_bound, vel_bound)   
        N_init = N_bound
        N_bound = len(pos_bound)
        i+=1
        print(N_init, N_bound)
        print('Number of bound particles: {} iteration: {}'.format(N_bound, i))

    return pos_bound, vel_bound, N_bound


if __name__ == "__main__":

   
    snapname = sys.argv[1]
    out_name = sys.argv[2]
    nmax = int(sys.argv[3])
    lmax = int(sys.argv[4])
    rs = float(sys.argv[5])

    
    pos, vel, mass = reading_particles(snapname)
    print('Snapshot loaded')
    print(pos[0])
    print(vel[0])
    print(mass[0])
    pos_bound, vel_bound, N_bound = find_bound_particles(pos,
                                                         vel,
                                                         mass,
                                                         rs,
                                                         nmax,
                                                         lmax)
    print('Bound particles computed')
    
    mass_bound = np.ones(len(ids_bound))
    print(pos_bound[0])
    print(vel_bound[0])
    lmc_bound = np.array([pos_bound[:,0], pos_bound[:,1], pos_bound[:,2], 
                          vel_bound[:,0], vel_bound[:,1], vel_bound[:,2],
                          mass_bound]).T
    
    np.savetxt(out_name, lmc_bound)
    
    print('Done writing snapshot with satellite bounded particles')




