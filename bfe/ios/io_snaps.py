import numpy as np
import h5py
from bfe.ios.read_snap import load_snapshot as readsnap
from bfe.ios.gadget_reader import is_parttype_in_file
import bfe.ios.com as com
from pynbody.analysis._com import shrink_sphere_center as ssc

def reshape_matrix(matrix, nmax, lmax, mmax):
    col_matrix = np.zeros((nmax+1, lmax+1, mmax+1))
    counter = 0
    for n in range(nmax+1):
        for l in range(lmax+1):
            for m in range(0, l+1):
                col_matrix[n][l][m] = matrix[counter]
                counter +=1
    return col_matrix


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

def read_snap_coordinates(path, snap, N_halo_part, com_frame='host', galaxy='host', snapformat=3):
    """
    Returns the MW properties.
    
    Parameters:
    path : str
        Path to the simulations
    snap : name of the snapshot
    satellite : Boolean
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
    all_pos = readsnap(path+snap, snapformat, 'pos', 'dm')
    all_vel = readsnap(path+snap, snapformat, 'vel', 'dm')
    all_ids = readsnap(path+snap, snapformat, 'pid', 'dm')
    all_pot = readsnap(path+snap, snapformat, 'pot', 'dm')
    all_mass = readsnap(path+snap, snapformat, 'mass', 'dm')


    
    
    print("Loading MW particles and LMC particles")

    if galaxy == 'host':
        print("Loading host particle")
        pos, vel, ids, pot, mass = host_particles(all_pos, all_vel, all_ids, all_pot, all_mass, N_halo_part)

    elif galaxy == 'sat':
        print("Loading satellite particles")
        pos, vel, ids, pot, mass = sat_particles(all_pos, all_vel, all_ids, all_pot, all_mass, N_halo_part)

    if com_frame == 'host': 
        print("Computing coordinates in the hots's COM frame")
        disk_particles = is_parttype_in_file(path+snap+".hdf5", 'PartType2')

        if disk_particles == True:
            print("* Computing host COM using minimum of the disk potential with partype: {}".format("PartType2"))
            pos_disk = readsnap(path+snap, snapformat,  'pos', 'disk')
            vel_disk = readsnap(path+snap, snapformat,  'vel', 'disk')
            pot_disk = readsnap(path+snap, snapformat, 'pot', 'disk')
            pos_cm, vel_cm = com.com_disk_potential(pos_disk, vel_disk, pot_disk)
            del pos_disk
            del vel_disk
            del pot_disk
        else:
            print("* Computing host COM using shrinking sphere in  partype: {}".format("PartType1"))
            pos_cm = ssc(np.ascontiguousarray(pos, dtype=float), np.ascontiguousarray(mass, dtype=float), min_particles=1000, shrink_factor=0.9, starting_rmax=500, num_threads=2)
            pos_re_center = com.re_center(pos, np.array(pos_cm))
            vel_cm = com.vcom_in(pos_re_center, vel, mass, rin=5) # TODO make rin value an input parameter? 
            print("* Done Computing host COM using shrinking sphere in  partype: {}".format("PartType1"))


    elif com_frame == 'sat':
        print('Computing coordinates in the satellite COM frame')
        if galaxy == 'host':
            LMC_pos, LMC_vel, LMC_ids, LMC_pot, LMC_mass = sat_particles(all_pos, all_vel, all_ids, all_pot, all_mass, N_halo_part)
            pos_cm, vel_cm  = com.CM(LMC_pos, LMC_vel, LMC_mass)
        else:
            # TODO: organize this method in a function
            # 
            # Guess com to re-center halo and precisely compute the COM
            rlmc = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
            #truncate = np.where(rlmc < 600)[0]
            # First COM guess
            pos1 = np.copy(pos)
            vel1 = np.copy(vel)

            #com1 = com.COM(pos1[truncate], vel1[truncate], np.ones(len(pos[truncate])))
            com1 = com.com(pos1, vel1, np.ones(len(pos)))
            pos_recenter = com.re_center(pos1, com1[0])
            vel_recenter = com.re_center(vel1, com1[1])

            rlmc = np.sqrt(pos_recenter[:,0]**2 + pos_recenter[:,1]**2 +  pos_recenter[:,2]**2)
            truncate = np.where(rlmc < 200)[0]
            
            com2 = com.com(
                pos_recenter[truncate], vel_recenter[truncate], 
                np.ones(len(truncate))*mass[0])

            pos_recenter2 = com.re_center(pos_recenter, com2[0])
            vel_recenter2 = com.re_center(vel_recenter, com2[1])

            rlmc = np.sqrt(pos_recenter2[:,0]**2 + pos_recenter2[:,1]**2 + pos_recenter2[:,2]**2)
            truncate2 = np.where(rlmc < 50)[0]
            
            #com3 = com.shrinking_sphere(
            #    pos_recenter2[truncate2], vel_recenter2[truncate2], 
            #    np.ones(len(truncate2))*mass[0])
            
            com3 = ssc(np.ascontiguousarray(pos_recenter2[truncate2], dtype=float), np.ascontiguousarray(np.ones(len(truncate2))*mass[0], dtype=float), min_particles=1000, shrink_factor=0.9, starting_rmax=500, num_threads=2)
            pos_recenter2_com = com.re_center(pos_recenter2[truncate2], np.array(com3))
            vcom3 = com.vcom_in(pos_recenter2_com, vel_recenter2[truncate2], np.ones(len(truncate2))*mass[0], rin=5) # TODO make rin value an input parameter? 


            pos_recenter3 = com.re_center(pos_recenter2, np.array(com3))
            vel_recenter3 = com.re_center(vel_recenter2, vcom3)
            
            rlmc = np.sqrt(pos_recenter3[:,0]**2 + pos_recenter3[:,1]**2 + pos_recenter3[:,2]**2)
            truncate3 = np.where(rlmc < 20)[0]
            
            com4 = ssc(np.ascontiguousarray(pos_recenter3[truncate3], dtype=float), np.ascontiguousarray(np.ones(len(truncate3))*mass[0], dtype=float), min_particles=1000, shrink_factor=0.9, starting_rmax=500, num_threads=2)

            pos_recenter3_com = com.re_center(pos_recenter3[truncate3], np.array(com4))
            vcom4 = com.vcom_in(pos_recenter3_com, vel_recenter3[truncate3], np.ones(len(truncate3))*mass[0], rin=5) # TODO make rin value an input parameter? 
            print(com1)
            print(com2)
            print(com3)
            print(com4)
            pos_cm = com1[0] + com2[0] + np.array(com3) + np.array(com4)
            vel_cm = com1[1] + com2[1] + vcom3 + vcom4
            print(pos_cm, vel_cm)

    elif com_frame == 'LSR' :
        print('Computing coordinates in the LSR frame')
        if disk_particles == True:
            print("Computing host COM with partype: {}".format("PartType2"))
            pos_disk = readsnap(path+snap, snapformat, 'pos', 'disk')
            vel_disk = readsnap(path+snap, snapformat, 'vel', 'disk')
            pot_disk = readsnap(path+snap, snapformat, 'pot', 'disk')
            pos_cm, vel_cm = com.com_disk_potential(pos_disk, vel_disk, pot_disk)
            del pos_disk
            del vel_disk
            del pot_disk
        else:
            print("Computing host COM with partype: {}".format("PartType1"))
            pos_cm, vel_cm = com.shrinking_sphere(pos, vel, mass)

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
    return pos_new, vel_new, pot, mass, ids, pos_cm, vel_cm
    


def write_coefficients(filename, results, nmax, lmax, r_s, mass, rcom ,vcom):
    """
    Coefficients file format.
    """

    Nrows = (nmax+1)*(lmax+1)*(lmax+1)
    
    ndim = np.shape(results)[1]
    
    if ndim == 2:
        Snlm = results[:,0]
        Tnlm = results[:,1]
        data = np.array([Snlm, Tnlm]).T
    elif ndim == 5:
        Snlm = results[:,0]
        Tnlm = results[:,1]
        varSnlm = results[:,2]
        varTnlm = results[:,3]
        varSTnlm = results[:,4]
        data = np.array([Snlm, varSnlm, Tnlm, varTnlm, varSTnlm]).T

    header = ' nmax: {:d} \n lmax: {:d} \n r_s: {:3.2f} \n particle_mass: {:10.3e} \n rcom: {} \n vcom: {}'.format(nmax, lmax, r_s, mass, rcom, vcom)

    np.savetxt(filename, data, header=header)

def write_coefficients_hdf5(filename, coefficients, exp_length, exp_constants, rcom):
    """
    Write coefficients into an hdf5 file

    """
    ndim = np.shape(coefficients)[1]
    nmax = exp_length[0]
    lmax = exp_length[1]
    mmax = exp_length[2]
    
    print("* Writing coefficients in {}".format(filename))
    
    rs = exp_constants[0]
    pmass = exp_constants[1]
    G = exp_constants[2]

    hf = h5py.File(filename + ".hdf5", 'w')
    print(coefficients[:,0])
    S = reshape_matrix(coefficients[:,0], nmax, lmax, mmax)
    T = reshape_matrix(coefficients[:,1], nmax, lmax, mmax)
    hf.create_dataset('Snlm', data=S, shape=(nmax+1, lmax+1, mmax+1))
    hf.create_dataset('Tnlm', data=T, shape=(nmax+1, lmax+1, mmax+1))
    hf.create_dataset('rcom', data=rcom, shape=(1, 3))
    hf.create_dataset('nmax', data=nmax)
    hf.create_dataset('lmax', data=lmax)
    hf.create_dataset('mmax', data=mmax)
    hf.create_dataset('rs', data=rs)
    hf.create_dataset('pmass', data=pmass)
    hf.create_dataset('G', data=G)
    if ndim == 5:
        Svar = reshape_matrix(coefficients[:,2], nmax, lmax, mmax)
        Tvar = reshape_matrix(coefficients[:,3], nmax, lmax, mmax)
        STvar = reshape_matrix(coefficients[:,4], nmax, lmax, mmax)
        hf.create_dataset('var_Snlm', data=Svar, shape=(nmax+1, lmax+1, mmax+1))
        hf.create_dataset('var_Tnlm', data=Tvar, shape=(nmax+1, lmax+1, mmax+1))
        hf.create_dataset('var_STnlm', data=STvar, shape=(nmax+1, lmax+1, mmax+1))
    hf.close()


def read_coefficients(filename):
    """
    Write coefficients into an hdf5 file

    """
    hf=h5py.File(filename + ".hdf5", 'r')
    print(hf.keys())
    print("* Loading coefficients")
    Snlm = np.array(hf.get('Snlm'))
    Tnlm = np.array(hf.get('Tnlm'))
    nmax = np.array(hf.get('nmax'))
    lmax = np.array(hf.get('lmax'))
    mmax = np.array(hf.get('mmax'))
    rs = np.array(hf.get('rs'))
    pmass = np.array(hf.get('pmass'))
    G = np.array(hf.get('G'))
    rcom = np.array(hf.get('rcom'))
    coefficients = [Snlm, Tnlm]
    if 'var_Snlm' in hf.keys():
        var_Snlm = np.array(hf.get('var_Snlm'))
        coefficients.append(var_Snlm)
    elif 'var_Tnlm' in hf.keys():
        var_Tnlm = np.array(hf.get('var_Tnlm'))
        coefficients.append(var_Tnlm)
    elif 'var_STnlm' in hf.keys():
        var_STnlm = np.array(hf.get('var_STnlm'))
        coefficients.append(var_STnlm)
    hf.close()

    return coefficients, [nmax, lmax, mmax], [rs, pmass, G], rcom
