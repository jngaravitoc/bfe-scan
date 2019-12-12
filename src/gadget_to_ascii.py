"""
Code to write ascii snapshots from Gadget binary 2 format.

to-do:
======

- Use bound particles of the host to compute the expansion?

"""

import numpy as np
import com
import sys

sys.path.append("../")
import reading_snapshots as rs



def truncate_halo(pos, vel, mass, ids, rcut):
    r_halo = (pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)**0.5
    rcut_index = np.where(r_halo<rcut)[0]
    return pos[rcut_index], vel[rcut_index], mass[rcut_index], ids[rcut_index]

def sample_halo(pos, vel, mass, npart_sample):
    
    n_halo_part = len(pos)
    N_random = np.random.randint(0, n_halo_part, npart_sample)
    	
    mass_fraction = n_halo_part/npart_sample		
    part_mass = mass*mass_fraction
    print('Particle mass factor', mass_fraction)
    print('New particle mass', part_mass)
    return pos[N_random], vel[N_random], part_mass

def npart_satellite(pos_sat, vel_sat, ids_sat, pmass_sat, pmass_host):
    """
    Sample satellite galaxies to have the same mass of the host satellite.
    """

    # number of particles in satellite
    init_sat_part = len(pos_sat)
    # Satellite total mass
    sat_tot_mass = pmass_sat*init_sat_part
    # new number of particles of satellite
    n_part_sat = int(sat_tot_mass/pmass_host)
    # new particles mass
    print('Initial number of satellite particles: ', init_sat_part)
    print('Final number of satellite particles after sampling', n_part_sat)
    rand = np.random.randint(0, init_sat_part, n_part_sat)
    # new particles mass
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


if __name__ == "__main__":
    path = "../../../MW_anisotropy/code/test_snaps/"
    #path = '/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b/mwlmc_raw/'
    #out_path_MW = '/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b/mwlmc_ascii/MW/'
    #out_path_LMC = '/media/ngaravito/4fb4fd3d-1665-4892-a18d-bdbb1185a07b/mwlmc_ascii/LMC/'
    out_path_MW = './'
    out_path_LMC = './'
    
    snap_names = ["MWLMC3_100M_new_b0_090", "MWLMC3_100M_new_b1_091",  
                  "MWLMC4_100M_new_b0_114", "MWLMC4_100M_new_b1_115",  
                  "MWLMC5_100M_new_b1_110", "MWLMC5_100M_new_b0_109",  
                  "MWLMC6_100M_new_b0_2_113", "MWLMC6_100M_new_b1_2_114"]



    n_halo_part = 100000000
    n_part_sample = 100000000
    #n_part_sample_sat = 1000000
    rcut_halo = 400
    sample = 0
    sample_lmc = 0
    #for i in range(0, len(snap_names)):
    for i in range(1, 8):
        halo = rs.read_snap_coordinates(path, snap_names[i], n_halo_part, com_frame='MW', galaxy='MW')
        # read_snap_coordinates returns pos, vel, pot, mass
        pos_halo_tr, vel_halo_tr, mass_tr, ids_tr = truncate_halo(halo[0], halo[1], halo[3], halo[4], rcut_halo)

        #satellite = rs.read_snap_coordinates(path, snap_names[i], n_halo_part, com_frame='sat', galaxy='sat')
        
        print("**************************")
        print(snap_names[i])
        #print(pos_cm, vel_cm)
        print("**************************")

        #pos_sat_tr, vel_sat_tr, mass_sat_tr, ids_sat_tr = truncate_halo(satellite[0], satellite[1], satellite[3], satellite[4], rcut_halo)
        
        #f sample == 1:
        #   pos_sample, vel_sample, mass_sample = sample_halo(pos_halo_tr, vel_halo_tr, halo[3][0], n_halo_part, n_part_sample)
        #lif sample == 0:
        #   mass_sample = mass_tr 
        #   pos_sample = pos_halo_tr
        #   vel_sample = vel_halo_tr

        pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em = npart_satellite(pos_sat_tr, vel_sat_tr, ids_sat_tr, mass_sat_tr[0], mass_tr[0])	
   
        # Outs: 
        #out_snap_host = 'MW_{}_{}'.format(int(len(pos_halo_tr)/1E6), snap_names[i])
        out_snap_sat= 'LMC_{}_{}'.format(int(len(pos_sat_em)/1E6), snap_names[i])
        #out_snap_sat= 'LMC_{}_{}'.format(int(len(satellite[0])/1E6), snap_names[i])

        #write_log([n_halo_part, halo[3][0], len(pos_sample), mass_sample], [len(pos_sat_tr[0]), satellite[3][0], len(pos_sat_em), mass_sat_em])
        #write_snap_txt(out_path_MW, out_snap_host, pos_halo_tr, vel_halo_tr, mass_tr, ids_tr)
        write_snap_txt(out_path_LMC, out_snap_sat, pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em)
        #write_snap_txt(out_path_LMC, out_snap_sat, satellite[0], satellite[1], satellite[3], satellite[4])
        
