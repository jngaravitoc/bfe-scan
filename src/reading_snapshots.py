import numpy as np
from pygadgetreader import readsnap
import com


def host_particles(xyz, vxyz, pids, pot, mass, N_host_particles):
    """
    Function that return the host and the sat particles
    positions and velocities.

    Parameters:
    -----------
    xyz: snapshot coordinates with shape (n,3)
    vxys: snapshot velocities with shape (n,3)
    pids: particles ids
    Nhost_particles: Number of host particles in the snapshot
    
    Returns:
    --------
    xyz, vxyz, ids, pot, mass.

    """
    sort_indexes = np.sort(pids)
    N_cut = sort_indexes[N_host_particles]
    host_ids = np.where(pids<N_cut)[0]
    return xyz[host_ids], vxyz[host_ids], pids[host_ids], pot[host_ids], mass[host_ids]


def sat_particles(xyz, vxyz, pids, pot, mass, Nhost_particles):
    """
    Function that return the host and the sat particles
    positions and velocities.

    Parameters:
    -----------
    xyz: snapshot coordinates with shape (n,3)
    vxys: snapshot velocities with shape (n,3)
    pids: particles ids
    Nhost_particles: Number of host particles in the snapshot
    Returns:
    --------
    xyz, vxyz, ids, pot, mass.

    """
    sort_indexes = np.sort(pids)
    N_cut = sort_indexes[Nhost_particles]
    sat_ids = np.where(pids>=N_cut)[0]
    return xyz[sat_ids], vxyz[sat_ids], pids[sat_ids], pot[sat_ids], mass[sat_ids]

def read_snap_coordinates(path, snap, N_halo_part, com_frame='MW', galaxy='MW'):
    """
    Returns the MW properties.
    
    Parameters:
    path : str
        Path to the simulations
    snap : name of the snapshot
    LMC : Boolean
        True or False if LMC is present on the snapshot.
    N_halo_part : int
        Number of particles in the MW halo.
    pot : boolean
        True or False if you want the potential back.
    com_frame : str
        Where the coordinates will be centered galactocentric (MW), on the
        satellite (sat), or in the LSR (LSR)
    galaxy : str
        galaxy coordinates to be returned (MW) or (sat)
    Returns:
    --------
    MWpos : 
    MWvel : 
    MWpot : 
    MWmass : 
    
    """
    # Load data
    print("Loading snapshot: "+path+snap)
    all_pos = readsnap(path+snap, 'pos', 'dm')
    all_vel = readsnap(path+snap, 'vel', 'dm')
    all_ids = readsnap(path+snap, 'pid', 'dm')
    all_pot = readsnap(path+snap, 'pot', 'dm')
    all_mass = readsnap(path+snap, 'mass', 'dm')


    
    
    print("Loading MW particles and LMC particles")

    if galaxy == 'MW':
        print('Loading MW particles')
        pos, vel, ids, pot, mass = host_particles(all_pos, all_vel, all_ids, all_pot, all_mass, N_halo_part)

    elif galaxy == 'sat':
        print('Loading satellite particles')
        pos, vel, ids, pot, mass = sat_particles(all_pos, all_vel, all_ids, all_pot, all_mass, N_halo_part)

    if com_frame == 'MW': 
        print('Computing coordinates in the MW COM frame')
        pos_disk = readsnap(path+snap, 'pos', 'disk')
        vel_disk = readsnap(path+snap, 'vel', 'disk')
        pot_disk = readsnap(path+snap, 'pot', 'disk')
        pos_cm, vel_cm = com.com_disk_potential(pos_disk, vel_disk, pot_disk)
        
    elif com_frame == 'sat':
        print('Computing coordinates in the satellite COM frame')
        if galaxy == 'MW':
            LMC_pos, LMC_vel, LMC_ids, LMC_pot, LMC_mass = sat_particles(all_pos, all_vel, all_ids, all_pot, all_mass, N_halo_part)
            pos_cm, vel_cm  = com.CM(LMC_pos, LMC_vel, LMC_mass)
        else:
            # Guess com to re-center halo and precisely compute the COM
            rlmc = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
            truncate = np.where(rlmc < 100)[0]
            # First COM guess
            com1 = com.COM(pos[truncate], vel[truncate], np.ones(len(truncate)))
            pos_recenter = com.re_center(pos[truncate], com1[0])
            print(com1[0])
            pos_cm, vel_cm  = com.CM(pos_recenter, vel[truncate], np.ones(len(mass[truncate]))*mass[0])
            print(pos_cm, vel_cm)
            pos_cm += com1[0]

    elif com_frame=='LSR':
        print('Computing coordinates in the LSR frame')
        pos_disk = readsnap(path+snap, 'pos', 'disk')
        vel_disk = readsnap(path+snap, 'vel', 'disk')
        pot_disk = readsnap(path+snap, 'pot', 'disk')
        pos_cm, vel_cm = com.com_disk_potential(pos_disk, vel_disk, pot_disk)

        pos_LSR = np.array([-8.34, 0, 0])
        vel_LSR = np.array([11.1,  232.24,  7.25])
        
        print(pos_LSR)
        print(vel_LSR)
    
    print(pos_cm, vel_cm)
    pos_new = com.re_center(pos, pos_cm)
    vel_new = com.re_center(vel, vel_cm)
    del pos
    del vel
    del all_pos
    del all_vel
    del all_pot
    del all_mass
    del all_ids
    del pos_disk
    del vel_disk
    del pot_disk
    return pos_new, vel_new, pot, mass, ids
    




