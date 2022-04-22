"""
Code to write ascii snapshots from Gadget binary 2 format.

TODO:
======


"""

import numpy as np

def truncate_halo(pos, vel, mass, ids, rcut):
    r_halo = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    rcut_index = np.where(r_halo<rcut)[0]
    return pos[rcut_index], vel[rcut_index], mass[rcut_index], ids[rcut_index]

def sample_halo(pos, vel, mass, npart_sample, ids):
    """
    Function that samples randomly a halo!
    it also corrects the mass of the new particle as follows:
        new particle mass = (old particle mass) * (N particles) / (N sample particle)

    return pos, vel, mass, ids
    """
    n_halo_part = len(pos)
    N_random = np.random.randint(0, n_halo_part, npart_sample)
    	
    mass_fraction = n_halo_part/npart_sample		
    part_mass = mass*mass_fraction
    print('Particle mass factor', mass_fraction)
    print('New particle mass', part_mass)
    return pos[N_random], vel[N_random], part_mass[N_random], ids[N_random]

def npart_satellite(pos_sat, vel_sat, ids_sat, pmass_sat, pmass_host):
    """
    Sample satellite galaxies to have the same mass of the host satellite.
    """

    # Number of particles in satellite
    init_sat_part = len(pos_sat)
    # Satellite total mass
    sat_tot_mass = pmass_sat*init_sat_part
    # New number of particles of satellite
    n_part_sat = int(sat_tot_mass/pmass_host)
    # New particles mass
    print('Initial number of satellite particles: ', init_sat_part)
    print('Final number of satellite particles after sampling', n_part_sat)
    rand = np.random.randint(0, init_sat_part, n_part_sat)
    # New particles mass
    new_part_mass = sat_tot_mass/n_part_sat
    return pos_sat[rand], vel_sat[rand], new_part_mass*np.ones(n_part_sat, dtype=float), ids_sat[rand]



def write_snap_txt(path, snap_name, pos, vel, mass, ids):
    np.savetxt(path+snap_name+'.txt', np.array([pos[:,0], pos[:,1], pos[:,2], 
                                                vel[:,0], vel[:,1], vel[:,2],
                                                mass, ids]).T)
    return 0

def write_log(halo, sat):
    """
    Printing summary
    """

    print('*********************************************')
    print('Summary:')
    print('Initial number of halo particles: ', halo[0])
    print('Initial halo particle mass: ', halo[1])
    print('Final number of halo particles', halo[2])
    print('Final halo particle mass', halo[3])
    print('Initial number of satellites particles: ', sat[0])
    print('Initial satellites particle mass: ', sat[1])
    print('Final number of satellites particles', sat[2])
    print('Final satellites particle mass', sat[3])
    print('*********************************************')

    return 0
    
#snap_names = ["MWLMC3_100M_new_b0_090", "MWLMC3_100M_new_b1_091",  
#                  "MWLMC4_100M_new_b0_114", "MWLMC4_100M_new_b1_115",  
#                  "MWLMC5_100M_new_b1_110", "MWLMC5_100M_new_b0_109",  
#                  "MWLMC6_100M_new_b0_2_113", "MWLMC6_100M_new_b1_2_114"] 
