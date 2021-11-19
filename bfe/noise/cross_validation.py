"""

Code to make a cross matching estimate of the coefficients and
the density of a system.

The code divide the system of N particles into N_batches with 
equal number of particles. Then it computes the coefficients
in each batch. 

With the coefficients in each batch the code computes the density 
for a given signal to noise (gamma) cut. 

The code assumes an assigns equal mass to each particle.
The mass of each particles in a batch is computed 
as:
    Np = 1/Nparticles_in_batch



Input:
------
    Positions of the particles : str
        filename with the positions of the particles in a numpy.array
        with size (Npart, 3).

    n_sample : int
        number of batches. Note, it has to be a integer of the total number of
        particles.
    r_s : float
        Hernquist Halo scale length
    nmax : int 
        order in the radial component of the expansion
    lmax : int
        order in the angular component of the expansion
    Npart_sample : int
        Number of particles to use from a halo. The division in batches
        would then be applied to this sample.

Returns:
-------
    Coefficients : 
        Files with the coeffients in each batch
    Densities:
        Files with 


Computes coefficients of
TODO:

    1. Make a stand alone parallel coefficient computation.
    2. Comment out the code
    3. Make the code as a stand alone script?
    4. Print coefficients and rho headers!
"""

import numpy as np
import schwimmbad
import bfe.ios.io_snaps as ios
import bfe.coefficients.parallel_coefficients as cop
import bfe.coefficients.parallel_potential as pop
import sys
import bfe.coefficients.coefficients_smoothing as coefficients_smoothing

def coeff_computation_all(pos, r_s, nmax, lmax, out_filename, cores=2):
    """
    Compute coefficients in parallel

    """
    npart_sample = np.shape(pos)[0]
    print(npart_sample)
    mass = np.ones(npart_sample)/npart_sample
    print("All particles mass:", mass[0])
    print('computing coefficients for all particles')
    pool = schwimmbad.choose_pool(mpi=False, processes=cores)
    batch = cop.Coeff_parallel(pos, mass, r_s, True, nmax, lmax)
    results = batch.main(pool)
    print('Done computing coefficients all')
    ios.write_coefficients(out_filename+'.txt', \
                           results, nmax, lmax,\
                           r_s, mass[0], rcom=0, vcom=0)


def random_halo_sample(n_batches, pos):
    """
    Make N bathes of random unique particles 3d positions.
    
    Parameters:
    ----------
        N batches : int
            Number of batches
        pos : numpy array
            Array with all the positions of the particles

    Returns:
    --------
        pos_shuffle: positions of the random particles shape(Nbatches, Npart_bath, 3)

    """
    Ninit = len(pos)
    
    assert Ninit%n_batches==0, 'N sample is not a multiple of the total number of particles'

        
    idx_random = np.arange(0, Ninit, 1)
    np.random.shuffle(idx_random)
    pos_rand = pos[idx_random]

    npart_sample = int(Ninit/n_batches)
    print("npart_sample: ",npart_sample)
    pos_shuffle = pos.reshape((n_batches, npart_sample, 3)) 
    return pos_shuffle




def coeff_computation(pos, r_s, nmax, lmax, out_filename, sn, cores=2):
    """
    Computes coefficients in N-batches

    """
    n_batches = np.shape(pos)[0]
    npart_sample = np.shape(pos)[1]
    mass = np.ones(npart_sample)/npart_sample
    for k in range(n_batches):    
        print('computing coefficients in batch {:d}'.format(k))
        pool = schwimmbad.choose_pool(mpi=False, processes=cores)
        batch = cop.Coeff_parallel(pos[k], mass, r_s, True, nmax, lmax)
        results = batch.main(pool)
        print('Done computing coefficients in batch {:d}'.format(k))
        ios.write_coefficients(out_filename+'_coefficients_batch_{:0>3d}.txt'.format(k), \
                               results, nmax, lmax,\
                               r_s, mass[0], rcom=0, vcom=0)
        rho_all = np.zeros((npart_sample, len(sn)+1))
        j=0
        for s in sn:
            pool_dens = schwimmbad.choose_pool(mpi=False, processes=cores)
            S = coefficients_smoothing.reshape_matrix(results[:,0], nmax, lmax, lmax)
            T = coefficients_smoothing.reshape_matrix(results[:,1], nmax, lmax, lmax)
            SS = coefficients_smoothing.reshape_matrix(results[:,2], nmax, lmax, lmax)
            TT = coefficients_smoothing.reshape_matrix(results[:,3], nmax, lmax, lmax)
            ST = coefficients_smoothing.reshape_matrix(results[:,4], nmax, lmax, lmax)
            print(np.shape(S))
            Ssmooth, Tsmooth, N = coefficients_smoothing.smooth_coeff_matrix(S, T, SS, TT, ST, mass[0],
                                   nmax, lmax, lmax, s)
            print(len(Ssmooth))
            batch_dens = pop.PBFEpot(pos[k], Ssmooth.flatten(),  Tsmooth.flatten(), r_s, nmax, lmax, G=1, M=1)
            rho =  batch_dens.main(pool_dens)
            print(len(Ssmooth))
            rho_all[:,j] = rho 
            j+=1
        np.savetxt(out_filename + "_rho_batch_{:0>3d}.txt".format(k), rho_all)
    return 0

if __name__ == "__main__":
    filename = sys.argv[1]
    print(filename)
    print('**********')
    outname = sys.argv[2]
    nbatches = 100
    r_s = 40.85
    nmax = 20
    lmax = 5
    data = np.loadtxt(filename)
    pos = data[:,0:3]
    Npart_sample = 1000000
    ## This lines is to sample a high resolution halo! 
    pos_rand = np.random.randint(0, len(pos), Npart_sample)
    sn = np.arange(0, 11, 0.2)
    coeff_computation_all(pos[pos_rand], r_s, nmax, lmax, outname)
    # Particles per batch are automatically found
    pos_batches = random_halo_sample(nbatches, pos[pos_rand])
    coeff_computation(pos_batches, r_s, nmax, lmax, outname, sn)
