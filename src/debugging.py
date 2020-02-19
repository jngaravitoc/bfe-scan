import numpy as np
import schwimmbad
import LMC_bounded as lmcb
import io_snaps as ios
import allvars
import gadget_to_ascii as g2a

def load_satellite_data(path, snapname, n_halo_part, npart_sample, 
                        sat_rs, nmax=15, lmax=8, ncores=4, rcut_halo=300):
    satellite = ios.read_snap_coordinates(path, 
                                          snapname, 
                                          n_halo_part, com_frame='sat', 
                                          galaxy='sat')
    pos_sat_tr, vel_sat_tr, mass_sat_tr, ids_sat_tr = g2a.truncate_halo(satellite[0], 
                                                                        satellite[1], 
                                                                        satellite[3], 
                                                                        satellite[4], 
                                                                        rcut_halo)
    N_sat_particles = len(mass_sat_tr)
    sample = np.random.randint(0, N_sat_particles , npart_sample)
    mass_fraction = N_sat_particles / npart_sample
    mass_sample = mass_sat_tr[sample]*mass_fraction
    print("particles masses in sample")
    print(mass_sample[0], mass_sat_tr[0], mass_fraction)
    return (pos_sat_tr[sample], vel_sat_tr[sample],
            mass_sample, ids_sat_tr[sample], 
            sat_rs, nmax, lmax, ncores) 
