##


import numpy as np
import matplotlib.pyplot as plt
from pygadgetreader import *



def all_host_particles(xyz, vxyz, pids, pot, mass, N_host_particles):
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
    xyz_mw, vxyz_mw, xyzlmc, vxyz_lmc: coordinates and velocities of
    the host and the sat.

    """
    sort_indexes = np.sort(pids)
    N_cut = sort_indexes[N_host_particles]
    host_ids = np.where(pids<N_cut)[0]
    return xyz[host_ids], vxyz[host_ids], pids[host_ids], pot[host_ids], mass[host_ids]


def host_sat_particles(xyz, vxyz, pids, pot, Nhost_particles, *p):
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
    xyz_mw, vxyz_mw, xyzlmc, vxyz_lmc: coordinates and velocities of
    the host and the sat.

    """
    sort_indexes = np.sort(pids)
    N_cut = sort_indexes[Nhost_particles]
    host_ids = np.where(pids<N_cut)[0]
    sat_ids = np.where(pids>=N_cut)[0]
    return xyz[host_ids], vxyz[host_ids], pids[host_ids], pot[host_ids], xyz[sat_ids], vxyz[sat_ids], pids[sat_ids], pot[sat_ids]



def sat_particles(xyz, vxyz, pids, pot, Nhost_particles):
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
    xyz_mw, vxyz_mw, xyzlmc, vxyz_lmc: coordinates and velocities of
    the host and the sat.

    """
    sort_indexes = np.sort(pids)
    N_cut = sort_indexes[Nhost_particles]
    host_ids = np.where(pids<N_cut)[0]
    sat_ids = np.where(pids>=N_cut)[0]
    return xyz[sat_ids], vxyz[sat_ids], pids[sat_ids], pot[sat_ids]




def COM(xyz, vxyz, m):
    """
    Returns the COM positions and velocities. 

    \vec{R} = \sum_i^N m_i \vec{r_i} / N
        
    """


    # Number of particles 
    N = sum(m)


    xCOM = np.sum(xyz[:,0]*m)/N
    yCOM = np.sum(xyz[:,1]*m)/N
    zCOM = np.sum(xyz[:,2]*m)/N

    vxCOM = np.sum(vxyz[:,0]*m)/N
    vyCOM = np.sum(vxyz[:,1]*m)/N
    vzCOM = np.sum(vxyz[:,2]*m)/N
    return [xCOM, yCOM, zCOM], [vxCOM, vyCOM, vzCOM]

def com_shrinking_sphere(pos, vel, m, delta=0.025):
    """
    Compute the center of mass coordinates and velocities of a halo
    using the Shrinking Sphere Method Power et al 2003.
    It iterates in radii until reach a convergence given by delta
    or 1% of the total number of particles.

    Parameters:
    -----------
    xyz: cartesian coordinates with shape (n,3)
    vxys: cartesian velocities with shape (n,3)
    delta(optional): Precision of the CM, D=0.025

    Returns:
    --------
    rcm, vcm: 2 arrays containing the coordinates and velocities of
    the center of mass with reference to a (0,0,0) point.

    """
        
    xCM = 0.0
    yCM = 0.0
    zCM = 0.0

    xyz = pos
    vxyz = vel

    N_i = len(xyz)
    N = N_i
        
    rCOM, vCOM = COM(xyz, vxyz, m)
    xCM_new, yCM_new, zCM_new = rCOM
    vxCM_new, vyCM_new, vzCM_new = vCOM
      


    while (((np.sqrt((xCM_new-xCM)**2 + (yCM_new-yCM)**2 + (zCM_new-zCM)**2) > delta) & (N>N_i*0.01)) | (N>1000)):
        xCM = xCM_new
        yCM = yCM_new
        zCM = zCM_new
        # Re-centering sphere
        R = np.sqrt((xyz[:,0]-xCM_new)**2 + (xyz[:,1]-yCM_new)**2 + (xyz[:,2]-zCM_new)**2)
        Rmax = np.max(R)
        # Reducing Sphere by its 2.5%
        index = np.where(R<Rmax*0.75)[0]
        xyz = xyz[index]
        vxyz = vxyz[index]
        m = m[index]
        N = len(xyz)
        #Computing new CM coordinates and velocities
        rCOM, vCOM = COM(xyz, vxyz, m)
        xCM_new, yCM_new, zCM_new = rCOM
        vxCM_new, vyCM_new, vzCM_new = vCOM
        print(Rmax)
        
    #elif self.prop == 'vel':
    #    print('this is not implemented yet')
      
    return np.array([xCM_new, yCM_new, zCM_new]), np.array([vxCM_new, vyCM_new, vzCM_new])


def com_disk_potential(xyz, vxyz, Pdisk):
    V_radius = 2
    vx = vxyz[:,0]
    vy = vxyz[:,1]
    vz = vxyz[:,2]
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    min_pot = np.where(Pdisk==min(Pdisk))[0]
    x_min = x[min_pot]
    y_min = y[min_pot]
    z_min = z[min_pot]
    # This >2.0 corresponds to the radius in kpc of the particles that
    # I am taking into account to compute the CM
    avg_particles = np.where(np.sqrt((x-x_min)**2.0 + (y-y_min)**2.0 + (z-z_min)**2.0)<V_radius)[0]
    x_cm = sum(x[avg_particles])/len(avg_particles)
    y_cm = sum(y[avg_particles])/len(avg_particles)
    z_cm = sum(z[avg_particles])/len(avg_particles)
    vx_cm = sum(vx[avg_particles])/len(avg_particles)
    vy_cm = sum(vy[avg_particles])/len(avg_particles)
    vz_cm = sum(vz[avg_particles])/len(avg_particles)
    return np.array([x_cm, y_cm, z_cm]), np.array([vx_cm, vy_cm, vz_cm])

def re_center(vec, cm):
    """
    Re center a halo to its center of mass.
    """
    vec_new = np.copy(vec)
    vec_new[:,0] = vec[:,0] - cm[0]
    vec_new[:,1] = vec[:,1] - cm[1]
    vec_new[:,2] = vec[:,2] - cm[2]
    return vec_new


def read_MW_snap_com_coordinates(path, snap, LMC, N_halo_part, pot, **kwargs):
    """
    Returns the MW properties.
    
    Parameters:
    path : str
        Path to the simulations
    snap : name of the snapshot
    LMC : boolean
        True or False if LMC is present on the snapshot.
    N_halo_part : int
        NUmber of particles in the MW halo.
    pot : booean
        True or False if you want the potential back.
        
    Returns:
    --------
    MWpos : 
    MWvel : 
    MWpot : 
    

    """
    print('Reading host  snapshot')
    MW_pos = readsnap(path+snap, 'pos', 'dm')
    MW_vel = readsnap(path+snap, 'vel', 'dm')
    MW_ids = readsnap(path+snap, 'pid', 'dm')

    pos_disk = readsnap(path+snap, 'pos', 'disk')
    vel_disk = readsnap(path+snap, 'vel', 'disk')
    pot_disk = readsnap(path+snap, 'pot', 'disk')

    pos_cm, vel_cm = com_disk_potential(pos_disk, vel_disk, pot_disk)
    print('Host disk com:', pos_cm, vel_cm)

    if pot == 1:
        MW_pot = readsnap(path+snap, 'pot', 'dm')
    if LMC == 1:
        print("Loading MW particles and LMC particles")
        MW_pos, MW_vel, MW_ids, MW_pot, LMC_pos, LMC_vel, LMC_ids, LMC_pot = host_sat_particles(MW_pos, MW_vel, MW_ids, MW_pot, N_halo_part)                                   
    
    MW_pos_cm = re_center(MW_pos, pos_cm)
    MW_vel_cm = re_center(MW_vel, vel_cm)
    
    if 'LSR' in kwargs:
        pos_LSR = np.array([-8.34, 0, 0])
        vel_LSR = np.array([11.1,  232.24,  7.25])
        # Values from http://docs.astropy.org/en/stable/api/astropy.coordinates.Galactocentric.html
        MW_pos_cm = re_center(MW_pos_cm, pos_LSR)
        MW_vel_cm = re_center(MW_vel_cm, vel_LSR)
        
    assert len(MW_pos) == N_halo_part, 'something is wrong with the number of selected particles'

    if pot == 1:
        return MW_pos_cm, MW_vel_cm, MW_pot, MW_ids
    else:
        return MW_pos_cm, MW_vel_cm, MW_ids
    
    
def read_satellite_snap_com_coordinates(snap, LMC, N_halo_part, pot):
    """
    Returns the satellite properties.
    
    Parameters:
    path : str
        Path to the simulations
    snap : name of the snapshot
    LMC : boolean
        True or False if LMC is present on the snapshot.
    N_halo_part : int
        NUmber of particles in the MW halo.
    pot : booean
        True or False if you want the potential back.
        
    Returns:
    --------
    MWpos
    MWvel
    MWpot *

    """
    print('Reading satellite snapshot')
    MW_pos = readsnap(snap, 'pos', 'dm')
    MW_vel = readsnap(snap, 'vel', 'dm')
    MW_ids = readsnap(snap, 'pid', 'dm')

    #pos_disk = readsnap(snap, 'pos', 'disk')
    #vel_disk = readsnap(snap, 'vel', 'disk')
    #pot_disk = readsnap(snap, 'pot', 'disk')


    if pot == 1:
        MW_pot = readsnap(snap, 'pot', 'dm')
    
    MW_pos, MW_vel, MW_ids, MW_pot, LMC_pos, LMC_vel, LMC_ids, LMC_pot = host_sat_particles(MW_pos, MW_vel, MW_ids, MW_pot, N_halo_part)

    pos_cm, vel_cm = com_shrinking_sphere(LMC_pos, LMC_vel, np.ones(len(LMC_pos)), delta=0.025)
    LMC_pos_cm = re_center(LMC_pos, pos_cm)
    LMC_vel_cm = re_center(LMC_vel, vel_cm)
   
    print('Satellite COM found at', pos_cm)
    return LMC_pos_cm, LMC_vel_cm, LMC_ids



def read_satellite_com(snap, N_halo_part, pot):
    """
    Returns the MW properties.
    
    Parameters:
    path : str
        Path to the simulations
    snap : name of the snapshot
    LMC : boolean
        True or False if LMC is present on the snapshot.
    N_halo_part : int
        NUmber of particles in the MW halo.
    pot : booean
        True or False if you want the potential back.
        
    Returns:
    --------
    MWpos
    MWvel
    MWpot *

    """
    MWLMC_pos = readsnap(snap, 'pos', 'dm')
    MWLMC_vel = readsnap(snap, 'vel', 'dm')
    MWLMC_ids = readsnap(snap, 'pid', 'dm')  
    MWLMC_pot = readsnap(path+snap, 'pot', 'dm')
    
    print("Loading LMC particles")
    LMC_pos, LMC_vel, LMC_ids, LMC_pot = sat_particles(MWLMC_pos,
                                                       MWLMC_vel,
                                                       MWLMC_ids,
                                                       N_halo_part)
    
    LMC_pos_cm = re_center(LMC_pos, pos_cm)
    LMC_vel_cm = re_center(LMC_vel, vel_cm)
    

    return LMC_pos_cm, LMC_vel_cm, LMC_ids

        

