import numpy as np
import schwimmbad
import bfe.ios.io_snaps as ios
import bfe.coefficients.parallel_coefficients as cop
import sys

def coeff_computation_all(pos, r_s, nmax, lmax, out_filename, cores=2):
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
    Ninit = len(pos)
    
    assert Ninit%n_batches==0, 'N sample is not a multiple of the total number of particles'

        
    idx_random = np.arange(0, Ninit, 1)
    np.random.shuffle(idx_random)
    pos_rand = pos[idx_random]

    npart_sample = int(Ninit/n_batches)
    pos_shuffle = pos.reshape((n_batches, npart_sample, 3)) 
    return pos_shuffle




def coeff_computation(pos, r_s, nmax, lmax, out_filename, cores=2):
    n_batches = np.shape(pos)[0]
    npart_sample = np.shape(pos)[1]
    mass = np.ones(npart_sample)/npart_sample
    for k in range(n_batches):    
        print('computing coefficients in batch {:d}'.format(k))
        pool = schwimmbad.choose_pool(mpi=False, processes=cores)
        batch = cop.Coeff_parallel(pos[k], mass, r_s, True, nmax, lmax)
        results = batch.main(pool)
        print('Done computing coefficients in batch {:d}'.format(k))
        ios.write_coefficients(out_filename+'_batch_{:0>3d}.txt'.format(k), \
                               results, nmax, lmax,\
                               r_s, mass[0], rcom=0, vcom=0)
        
    return 0

if __name__ == "__main__":
    filename = sys.argv[1]
    print(filename)
    print('**********')
    outname = sys.argv[2]
    n_sample = 50
    r_s = 40.85
    nmax = 20
    lmax = 5
    data = np.loadtxt(filename)
    pos = data[:,0:3]
    pos_rand = np.random.randint(0, len(pos), 100000)
    coeff_computation_all(pos[pos_rand], r_s, nmax, lmax, outname)
    pos_batches = random_halo_sample(n_sample, pos[pos_rand])
    coeff_computation(pos_batches, r_s, nmax, lmax, outname)

