"""
TODO: 
    - Test speed of the code and compare with that one of the BFE-c
    - Write coefficients function in python or use cython ?
     
"""

import numpy as np
import schwimmbad 
from gala.potential.scf._computecoeff import STnlm_discrete, STnlm_var_discrete
from bfe.ios import read_snap
from bfe.ios import write_coefficients

class Coeff_parallel(object):
    def __init__(self, pos, mass, r_s, var, nmax, lmax):
        self.pos = pos
        self.mass = np.ascontiguousarray(mass).astype("float64")
        self.r_s = r_s
        self.var = var
        self.r = np.sqrt(np.sum(np.ascontiguousarray(self.pos)**2, axis=-1))
        self.s = np.ascontiguousarray(self.r / r_s).astype("float64")
        self.phi = np.arctan2(np.ascontiguousarray(self.pos[:,1]), np.ascontiguousarray(self.pos[:,0])).astype("float64")
        self.X = np.ascontiguousarray(self.pos[:,2] / self.r).astype("float64")
        self.nmax = nmax
        self.lmax = lmax
        print("* Computing SCF coefficients in parallel")

    def nlm_list(self, nmax, lmax):
        """
        
        """
        nlm = []
        for n in range(nmax+1):
            for l in range(lmax+1):
                for m in range(l+1):
                    nlm.append((n,l,m))                 
        return nlm
                                                                                            
    def compute_coeffs_discrete_parallel(self, task):
        n, l, m = task
        S, T = STnlm_discrete(self.s, self.phi, self.X, self.mass, n, l, m)

        if self.var == True:
            varS, varT, varST = STnlm_var_discrete(self.s, self.phi, self.X, self.mass, n, l, m)
            return S, T, varS, varT, varST
        else :
            return S, T

    def main(self, pool):
        tasks = self.nlm_list(self.nmax, self.lmax)
 
        results = pool.map(self.compute_coeffs_discrete_parallel, tasks)
        pool.close()

        return np.array(results)



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

if __name__ == "__main__":
    snap_name = '../../MW_anisotropy/code/test_snaps/LMC3_507K_part_b1_091.txt'
    pos, mass = read_snap.load_snapshot(snap_name, snapformat=2, masscol=6)
    nmax = 5
    lmax = 5
    r_s = 10.0
    print("Done loading data")
    pool = schwimmbad.choose_pool(mpi=False, processes=4)
    halo = Coeff_parallel(pos, mass, r_s, False, nmax, lmax)
    results = halo.main(pool)
    print(np.shape(results))
    Snlm, Tnlm = coeff_matrix(results)
    varSnlm = np.ones(len(results[:,0]))
    varTnlm = np.ones(len(results[:,0]))
    varSTnlm = np.ones(len(results[:,0]))
  
    write_coefficients("test_coefficients.txt", results, nmax, lmax, r_s, mass[0])

