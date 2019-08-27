"""
Code to write ascii snapshots from Gadget binary 2 format.

to-do:
======

- Same mass in satellite and host
- Use bound particles of the host to compute the expansion?

"""

import numpy as np
import pygadgetreader 
import reading_snapshots


def read_mw_com_snapshot(path, snap_name, n_halo_part):
    """
    Reads MW particles
    """
    pos, vel, pot, ids = reading_snapshots.read_MW_snap_com_coordinates(path, snap_name, LMC=True,
                                                                       N_halo_part=n_halo_part, pot=True)
    mass =  pygadgetreader.readsnap(path+snap_name, 'mass', 'dm')
    all_ids =  pygadgetreader.readsnap(path+snap_name, 'pid', 'dm')
    mw_mass = np.where(all_ids==ids[0])
    return pos, vel, mass[mw_mass]


def read_sat_com_snapshot(path, snap_name, n_halo_part):
    """
    Read Satellites particles
    """
    pos, vel, lmc_ids = reading_snapshots.read_satellite_snap_com_coordinates(path+snap_name, LMC=True,
                                                                       N_halo_part=n_halo_part, pot=True)
    mass =  pygadgetreader.readsnap(path+snap_name, 'mass', 'dm')
    ids =  pygadgetreader.readsnap(path+snap_name, 'pid', 'dm')
    lmc_mass = np.where(ids==lmc_ids[0])
        
    return pos, vel, mass[lmc_mass], len(ids)


def truncate_halo(pos, vel, rcut):
    r_halo = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    rcut_index = np.where(r_halo<rcut)[0]
    return pos[rcut_index], vel[rcut_index]

def sample_halo(pos, vel, mass, n_halo_part, npart_sample):
    assert(npart_sample<n_halo_part)

    N_random = np.random.randint(0, len(pos), npart_sample)
    	
    mass_fraction = n_halo_part/npart_sample		
    part_mass = mass*mass_fraction
    print('Particle mass factor', mass_fraction)
    print('New particle mass', part_mass)
    return pos[N_random], vel[N_random], part_mass

def npart_satellite(pos_sat, vel_sat,  pmass_sat, pmass_host):
    """
    Sample satellite galaxies to have the same mass of the host satellite.
    """

    # number of particles in satellite
    init_sat_part = len(pos_sat)
    # Satellite total mass
    sat_tot_mass = pmass_sat*init_sat_part
    # new number of particles of satellite
    n_part_sat = int(sat_tot_mass/pmass_host)
    # sampling satellite particles
    n_part_sample = np.random.randint(0, len(pos_sat), n_part_sat)
    # new particles mass
    new_part_mass = sat_tot_mass/n_part_sat
    return pos_sat[n_part_sample], vel_sat[n_part_sample], new_part_mass



def write_snap_txt(snap_name, pos, vel, mass):
    np.savetxt(snap_name+'.txt', np.array([pos[:,0], pos[:,1], pos[:,2], vel[:,0], vel[:,1], vel[:,2], mass*np.ones(len(pos))]).T)
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


if __name__ == "__main__":
    path = '../../MW_anisotropy/code/test_snaps/'
    
    #snap_name = 'MWLMC6_2_100M_new_b1_112' 
    snap_name = 'MWLMC5_100M_new_b1_110' 
    #snap_name = 'MWLMC4_100M_new_b1_115' 
    #snap_name = 'MWLMC3_100M_new_b1_091' 
    
    n_halo_part = 100000000
    n_part_sample = 10000000
    #n_part_sample_sat = 1000000
    rcut_halo = 400


    pos_halo, vel_halo, mass_halo = read_mw_com_snapshot(path, snap_name, n_halo_part)
    pos_halo_tr, vel_halo_tr = truncate_halo(pos_halo, vel_halo, rcut_halo)
    pos_satellite, vel_satellite, mass_satellite , n_sat = read_sat_com_snapshot(path, snap_name, n_halo_part)

    pos_sample, vel_sample, mass_sample = sample_halo(pos_halo_tr, vel_halo_tr, mass_halo, n_halo_part, n_part_sample)
    pos_sat_em, vel_sat_em, mass_sat_em =  npart_satellite(pos_satellite, vel_satellite, mass_satellite, mass_sample)	
    #pos_sat_sample, mass_sat_sample = sample_halo(pos_satellite, mass_satellite, n_sat, n_part_sample_sat)
   
    # Outs: 
    #out_snap_host = 'MWLMC3_{}M_part_b1_091'.format(int(len(pos_sample)/1E6))
    out_snap_sat= 'LMC5_{}K_part_b1_110'.format(int(len(pos_sat_em)/1E3))

    #write_log([n_halo_part, mass_halo, len(pos_sample), mass_sample], [len(pos_satellite), mass_satellite, len(pos_sat_em), mass_sat_em])
    #write_snap_txt(path+out_snap_host, pos_sample, vel_sample, mass_sample)
    write_snap_txt(path+out_snap_sat, pos_sat_em, vel_sat_em, mass_sat_em)
