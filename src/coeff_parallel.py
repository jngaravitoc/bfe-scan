"""
TODO: 
    - Test speed of the code and compare with that one of the BFE-c

"""

import numpy as np
import schwimmbad 
from gala.potential.scf._computecoeff import STnlm_discrete, STnlm_var_discrete
#import read_snap


class Coeff_parallel(object):
    def __init__(self, pos, mass, r_s, var):
        self.pos = pos
        self.mass = np.ascontiguousarray(mass).astype("float64")
        self.r_s = r_s
        self.var = var
        self.r = np.sqrt(np.sum(np.ascontiguousarray(self.pos)**2, axis=-1))
        self.s = np.ascontiguousarray(self.r / r_s).astype("float64")
        self.phi = np.arctan2(np.ascontiguousarray(self.pos[:,1]), np.ascontiguousarray(self.pos[:,0])).astype("float64")
        self.X = np.ascontiguousarray(self.pos[:,2] / self.r).astype("float64")

    def compute_coeffs_discrete_parallel(self, n, l, m):
        
        S, T = STnlm_discrete(self.s, self.phi, self.X, self.mass, n, l, m)

        if self.var == True:
            varS, varT, varST = STnlm_var_discrete(self.s, self.phi, self.X, self.mass, n, l, m)
            return S, T, varS, varT, varST
        else:
            return S, T

    def __call__(self, task):
        n, l, m = task 
        return self.compute_coeffs_discrete_parallel(n, l, m)


def nlm_list(nmax, lmax):
    """

    """
    nlm = []
    for n in range(nmax+1):
        for l in range(lmax+1):
            for m in range(l+1):
                nlm.append((n,l,m))                 
    return nlm
                                                                                            

def main(pool, pos, mass, nmax, lmax, r_s, var=True):
    worker = Coeff_parallel(pos, mass, r_s, var)

    tasks = nlm_list(nmax, lmax)
    
    results = pool.map(worker, tasks)
        
    pool.close()

    return results

def coeff_matrix(STnlm):
    Snlm_matrix = np.zeros((nmax+1, lmax+1, lmax+1))
    Tnlm_matrix = np.zeros((nmax+1, lmax+1, lmax+1))
    
    dim_ =  np.shape(STnlm)[1]
    if dim_ == 5:
        varSnlm_matrix = np.zeros((nmax+1, lmax+1, lmax+1))
        varTnlm_matrix = np.zeros((nmax+1, lmax+1, lmax+1))
        varSTnlm_matrix = np.zeros((nmax+1, lmax+1, lmax+1))
    
    k = 0
    
    for n in range(nmax+1):
        for l in range(lmax+1):
            for m in range(l+1):
                Snlm_matrix[n,l,m] = STnlm[k][0]
                Tnlm_matrix[n,l,m] = STnlm[k][1]
                if dim_ == 5:
                    varSnlm_matrix[n,l,m] = STnlm[k][2]
                    varTnlm_matrix[n,l,m] = STnlm[k][3]
                    varSTnlm_matrix[n,l,m] = STnlm[k][4]
                
                k+=1

    if dim_ == 5:
        return Snlm_matrix, varSnlm_matrix, Tnlm_matrix, varTnlm_matrix, varSTnlm_matrix
    else:
        return Snlm_matrix, Tnlm_matrix

def write_coefficients(filename, Snlm, varSnlm, Tnlm, varTnlm, varSTnlm, nmax, lmax, r_s):
    """
    
    """
    Nrows = (nmax+1)*(lmax+1)*(lmax+1)

    data = np.array([Snlm.flatten(), varSnlm.flatten(), Tnlm.flatten(),
                         varTnlm.flatten(), varSTnlm.flatten()]).T
    header = 'nmax: {:d}, lmax: {:d}, r_s: {:3.2f}'.format(nmax, lmax, r_s)
    np.savetxt(filename, data, header=header)

    

def compute_coeff_parallel(pos, nmax, lmax, r_s, ncores):
    """

    """
    #pool = schwimmbad.choose_pool(mpi=False, processes=ncores)
    pool = schwimmbad.choose_pool(mpi=True, processes=2)
    results = main(pool, pos, mass, nmax, lmax, r_s)
    Snlm, Tnlm = coeff_matrix(results)
    return Snlm, Tnlm

"""
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                        type=int, help="Number of processes (uses multiprocessing.)")
    group.add_argument("--mpi", dest="mpi", default=False,
                        action="store_true", help="Run with MPI.")

    args = parser.parse_args()
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)


    snap_name = '../../MW_anisotropy/code/test_snaps/LMC3_507K_part_b1_091.txt'
    pos, mass = read_snap.load_snapshot(snap_name, snapformat=2, masscol=6)
    nmax = 5
    lmax = 20
    r_s = 10
    
    results = main(pool, pos, mass, nmax, lmax, r_s, var=True)
    
    Snlm, varSnlm, Tnlm, varTnlm, varSTnlm = coeff_matrix(results)
    
    write_coefficients("test_coefficients.txt", Snlm, varSnlm, Tnlm, varTnlm, varSTnlm, nmax, lmax, r_s)
"""
