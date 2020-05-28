"""
Code to compute the GOF

"""

import numpy as np
import time, datetime
import scipy.special as special
import scipy.integrate as integrate
from joblib import Parallel, delayed
import sys
# Local libraries
sys.path.append('/home/u9/jngaravitoc/codes/Kullback-Leibler/')
import coefficients_smoothing


def time_now(): 
    h = datetime.datetime.now().hour
    m = datetime.datetime.now().minute
    s = datetime.datetime.now().second
    print("{}:{}:{}".format(h, m, s))
    return (h, m, s)

def time_diff(t1, t2):
    print("Time diff:", t2[0]-t1[0], t2[1]-t1[1], t2[2]-t1[2])
    return 0

def write_output(file_name, data):
    np.savetxt(file_name, data)

def Knl(n, l):
    """
    Knl definition see Equation 12 in Lowing+11 
    https://ui.adsabs.harvard.edu/abs/2011MNRAS.416.2697L/abstract

    """
    return ( 0.5*n*(n+4*l+3) )+( (l+1)*(2*l+1) )

def integrand(xi, n, n_p, l):
    """
    Integral to compute the first term in the gof of a BFE.

    \hat{\rho}(x|\gamma)^2

    See Equation 8 in companion notes. 
    """

    factor = (xi+1)**(2*l) * (1-xi)**(2*l+3/2.)
    gegenbauer1 = special.gegenbauer(n, 2*l+3/2.)
    gegenbauer2 = special.gegenbauer(n_p, 2*l+3/2.)
                           
    return factor*gegenbauer1(xi)*gegenbauer2(xi)


def factors(n, n_p, l, r_s):
    """
    Constants values of the integral in integrand.
    See Equation 8 in Companion notes.
    """

    f = 1/np.pi*r_s/2**(4*l+5)
    K_nl = Knl(n, l)
    K_npl = Knl(n_p, l)
    return f*K_nl*K_npl

def coefficients_sum(S, T, nmax, lmax, r_s):
    """
    Sums over each component: See equation 5 in the companion notes.
    """
    rho2 = 0
    for n_p in range(nmax):
      for n in range(nmax):
        for l in range(lmax):
          f = factors(n, n_p, l, r_s)
          I = integrate.quad(integrand, -1, 1, args=(n, n_p, l))[0]
          for m in range(lmax):
            rho2 += 2*(S[n,l,m]*S[n_p,l,m] + T[n,l,m]*T[n_p,l,m])*(-1)**m * f * I   
    return rho2



def coefficients_sum_fast(S, T, order, r_s):
    """
    Faster way to compute the sums over the coefficients. 
    Just takes into account the coefficients that are non-zero
    and skips the sum over all of the coefficients.

    """

    # Selects coefficients that are non-zero
    index = np.where((S**2+T**2)>0)
    #n_unique = np.unique(index[0])
    nmax = index[0]
    lmax = index[1]
    mmax = index[2]
    
    # initialize sum
    rho2 = 0
    #max_all = np.arange(0, nmax, 1)
    #max=lmax
    #_unique = nmax
   
    # I can't skip the sum over n_p
    for n_p in range(0, order):
      # Simplified sum over all of the coefficients that are non-zero 
      for i in range(len(nmax)):
        f = factors(nmax[i], n_p, lmax[i], r_s)
        I = integrate.quad(integrand, -1, 1, args=(nmax[i], n_p, lmax[i]))[0]
        rho2 += 2*(S[nmax[i],lmax[i],mmax[i]]*S[n_p,lmax[i],mmax[i]] + T[nmax[i],lmax[i],mmax[i]]*T[n_p,lmax[i],mmax[i]])*(-1)**mmax[i] * f * I 

    return rho2


def rho_square(i, sn):
    """
    
    Computes the `True' density square as defined 
    in equation 5 of the companion doc.
    
    It use the coefficients computed with the particle distribution 
    in each particle batch i. 

    """

    path = '/home/xzk/work/github/MW-LMC-SCF/code/KL/data/'
    filename = 'coeff_MW_100M_b1_dm_part_1e6_300'
    mass = 1/1E5
    #mass = 1.577212515257997438e-05
    coeff = coefficients_smoothing.read_coeffcov_matrix(path+filename,
                                                       1, nmax, lmax, lmax, i,
                                                       read_type=0)
    S = coeff[0]
    SS = coeff[2]
    T = coeff[1]
    TT = coeff[3]
    ST = coeff[4]
    S_smooth, T_smooth, N_coeff = coefficients_smoothing.smooth_coeff_matrix(S, T, SS, TT, ST, mass,
                                                                             nmax,lmax, lmax, sn_range[sn])
    
    rho2 = coefficients_sum(S_smooth, T_smooth, nmax, lmax, rs)
    return rho2
    
def sum_rho_true(i, sn):
    """
    Computes the second term in the Goodness of Fit computation
    using cross-validation.
    See equation 1 in the companion document.


    """

    rho = np.loadtxt(densities_file+'_{:0>3d}_sn_{:0>3d}.txt'.format(i, sn))
    rho_true = np.zeros(nbatches-1)
    k=0
    mass = 1/1000
    for j in range(nbatches):
      if i!=j:
        rho_true[k] = np.sum(mass*rho[:,j])
        k+=1
    return np.sum(rho_true)/(nbatches-1)

def Likelihood(sn):
    L = np.zeros(nbatches)
    L2 = np.zeros(nbatches)
    print('This is sn {}'.format(sn))
    #1 = 0
    L1 = rho_square(0, sn)
    #for i in range(nbatches):
        #L1 =  rho_square(i, sn)
        #ho_true = sum_rho_true(i, sn)
        #2[i] = -2 * rho_true 
        #[i] = L1 + L2[i]
    print('Done with batches of sn {}'.format(sn))

    return  L1


def coeff_rand_generator(order):
        # generates random coefficients with different orders
    S = np.random.random((order, order, order))
    T = np.random.random((order, order, order))
    return S, T



if __name__ == "__main__":
    #densities_file = "/extra/jngaravitoc/KL_data/MW_100M_b1_dm_part_1e6_300_KL_analysis_batch"
    #ensities_file='/rsgrps/gbeslastudents/nicolas/KL_data/MW_100M_b1_dm_part_1e6_300_KL_analysis_batch'
    file_name = "/home/xzk/work/github/MW-LMC-SCF/code/KL/data/gof_rho_true.txt"
    npart = 1000
    #ass = 1 / npart
    nbatches = 100
    sn_range = np.arange(0, 11, 0.2)
    num_cores = 13
    nmax = 20
    lmax = 5
    rs = 40.85
    output = Parallel(n_jobs=num_cores)(delayed(Likelihood)(sn) for sn in range(4, 50))
    #print(output)
    write_output(file_name, output)
    
    #order = int(sys.argv[1])
    #S, T = coeff_rand_generator(order)
    #t1 = time.clock()
    #rho1 = coefficients_sum_fast(S, T, order, 10)
    #t2 = time.clock()
    #rho2 = coefficients_sum(S, T, order, order, 10)
    #t3 = time.clock()
    #dt1 = t2-t1
    #dt2 = t3-t2
    #print('fast:', dt1)
    #print('slow:', dt2)
    #print(rho1, rho2)
