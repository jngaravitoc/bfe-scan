"""
Script to found the bound particles of a satellite
galaxy.

author: Nicolás Garavito-Camargo

usage: python LMC_bounded.py filename nmax lmax rs

"""


import numpy as np
import schwimmbad
import gala.potential as gp
import coeff_parallel as cop


class Pot_parallel(object):
    def __init__(self, S, T, r_s):
        self.S = S
        self.T = T
        self.r_s = r_s


    def compute_scf_pot(self, pos):
        """
        TODO: Parallelize this function here!
        """
        G_gadget = 43007.1
        LMC_potential = gp.potential(self.S, self.T, M=1, r_s=self.r_s, G=G_gadget)
        potential = LMC_potential(np.ascontiguousarray(pos).astype(float))
        return potential

    def __call__(self, task):
        return self.compute_scf_pot(task)




def compute_scf_pot(pos, S, T, rs):
    """
    TODO: Parallelize this function here!
    """
    G_gadget = 43007.1
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





