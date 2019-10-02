import numpy as np
from numpy import linalg



## Reading coefficients
def reshape_matrix(matrix, n, l, m):
    col_matrix = np.zeros((n+1, l+1, m+1))


    counter = 0
    for n in range(n+1):
        for l in range(l+1):
            for m in range(0, l+1):
                col_matrix[n][l][m] = matrix[counter]
                counter +=1
    return col_matrix
    

def read_coeff_matrix(filename, nfiles, n, l, m, n_min=0, n_max=1000, snaps=0):
    """
    Compute the mean of the coefficients from multiple files and return the mean values.
    """
    assert(nfiles>=(n_max-n_min))
    S_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    T_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    
    S_mean = np.zeros((int((n+1)*(l+1)*(l/2.+1))))
    T_mean = np.zeros((int((n+1)*(l+1)*(l/2.+1))))

    for i in range(n_min, n_max):
        if snaps==0:
            #coeff = np.loadtxt(filename + '{:03d}.txt'.format(i))
            coeff = np.loadtxt(filename + '{:03d}_snap_0000.txt'.format(i))
        elif snaps==1:
            coeff = np.loadtxt(filename + '000_snap_{:04d}.txt'.format(i))

        S_matrix[:,i-n_min] = coeff[:,0]
        T_matrix[:,i-n_min] = coeff[:,1]
        
    for i in range(len(S_matrix[:,0])):
        S_mean[i] = np.mean(S_matrix[i])
        T_mean[i] = np.mean(T_matrix[i])

    S_mean_matrix = reshape_matrix(S_mean, n, l, m)
    T_mean_matrix = reshape_matrix(T_mean, n, l, m)

    return S_mean_matrix, T_mean_matrix
    
def read_cov_elements(filename, nfiles, n, l, m, n_min=0, n_max=1000, snaps=0):


    Scov_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    Tcov_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    STcov_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    
    S_mean_cov_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1))))
    T_mean_cov_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1))))
    ST_mean_cov_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1))))

    for i in range(n_min, n_max):
    
        if snaps==0:
            cov = np.loadtxt(filename + '{:03d}_snap_0000.txt'.format(i))
            #cov = np.loadtxt(filename + '{:03d}.txt'.format(i))
        elif snaps==1:
            cov = np.loadtxt(filename + '0000_snap_{:04d}.txt'.format(i))



        Scov_matrix[:,i-n_min] = cov[:,0]
        Tcov_matrix[:,i-n_min] = cov[:,1]
        STcov_matrix[:,i-n_min] = cov[:,2]
        
    for i in range(len(Scov_matrix[:,0])):
        S_mean_cov_matrix[i] = np.mean(Scov_matrix[i])
        T_mean_cov_matrix[i] = np.mean(Tcov_matrix[i])        
        ST_mean_cov_matrix[i] = np.mean(STcov_matrix[i])
   
    S_mean_cov_matrix = reshape_matrix(S_mean_cov_matrix, n, l, m)
    T_mean_cov_matrix = reshape_matrix(T_mean_cov_matrix, n, l, m)     
    ST_mean_cov_matrix = reshape_matrix(ST_mean_cov_matrix, n, l, m)     
               
    return S_mean_cov_matrix, T_mean_cov_matrix, ST_mean_cov_matrix


def covariance_matrix_builder(S, T, SS, TT, ST, mass):
    """

    """
    cov_matrix = np.zeros((2,2))
    cov_matrix[0][0] = SS - mass*S**2
    cov_matrix[0][1] = ST - mass*S*T
    cov_matrix[1][1] = TT - mass*T**2
    cov_matrix[1][0] = cov_matrix[0][1]

    return cov_matrix
    
    
def smoothing(S, T, varS, varT):
    """
    Computes optimal smoothing following Eq.8 in Weinberg+96.
    
    returns:
    --------
    
    bs
    bt : 
    """
    bs = 1 / (1 + (varS/S**2))
    bt = 1 / (1 + (varT/T**2))
    if S == 0:
        bs=0
    if T == 0:
        bt=0
    return bs, bt

def smoothing_coeff_uncorrelated(cov_matrix, S, T, sn=0, verb=False):
    # SVD decomposition of the covariance matrix
    T_rot, v, TL = linalg.svd(cov_matrix)
    
    # Computes inverted transformation matrix
    T_rot_inv = linalg.inv(T_rot)

    # Variances of the coefficients in the uncorrelated base.
    varS = v[0]
    varT = v[1]
    
    ## uncorrelated coefficients
    coeff_base = np.array([S, T])
    S_unc, T_unc = np.dot(T_rot, coeff_base)
    b_S_unc, b_T_unc = smoothing(S_unc, T_unc, varS, varT)
    S_unc_smooth = S_unc*b_S_unc
    T_unc_smooth = T_unc*b_T_unc
    SN_coeff_unc = (S_unc**2/varS)**0.5
        
    S_smooth, T_smooth = np.dot(T_rot_inv, np.array([S_unc_smooth, T_unc_smooth]))
    
    if verb==True:
        print("S,T  correlated = ", S, T)
        print("S,T uncorrelated = ", S_unc, T_unc)
        print("Uncorrelated smoothing = ", b_S_unc, b_T_unc)
        print("S, T uncorrelated smoothed =", S_unc_smooth, T_unc_smooth)
    
    n=1
    if SN_coeff_unc < sn:
        S_smooth = 0
        T_smooth = 0
        n=0
    
            
    return S_smooth, T_smooth, n
    
def smooth_coeff(S, T, SS, TT, ST, mass, verb=False, sn=0):
    cov_matrix = covariance_matrix_builder(S, T, SS, TT, ST, mass)
    S_smooth, T_smooth, n_coeff = smoothing_coeff_uncorrelated(cov_matrix, S, T, sn, verb)
    
    return S_smooth, T_smooth, n_coeff

def smooth_coeff_matrix(S, T, SS, TT, ST, mass, nmax, lmax, mmax, sn):
    S_matrix_smooth = np.zeros((nmax+1, lmax+1, lmax+1))
    T_matrix_smooth = np.zeros((nmax+1, lmax+1, lmax+1))
    n_coefficients = 0
    for n in range(nmax+1):
        for l in range(lmax+1):
            for m in range(l+1):
                S_matrix_smooth[n][l][m], T_matrix_smooth[n][l][m], n_coeff = smooth_coeff(S[n][l][m],
                                                                                  T[n][l][m], 
                                                                                  SS[n][l][m], 
                                                                                  TT[n][l][m],
                                                                                  ST[n][l][m],
                                                                                  mass, verb=False, 
                                                                                  sn=sn)
                n_coefficients += n_coeff
    return S_matrix_smooth, T_matrix_smooth, n_coefficients
    
 
def smoothing_biased(cov_matrix, coeff, m, sn):
    """
    Coefficients smoothing 
    
    """
    var_coeff = ((cov_matrix - m*coeff**2))
    b_nlm = 1/(1 + var_coeff/coeff**2)
    #if coeff == 0:
    #    b_nlm=0
    #if b_nlm <= 0.3:
    #    b_nlm=0
    # This line remove nans an put the original values
    b_nlm_values = np.nan_to_num(b_nlm)
    
    SN_coeff_unc = (coeff**2/var_coeff)**0.5
    
    n=1
    if SN_coeff_unc < sn:
        b_nlm_values=0.0
        n=0 
    
    
    return b_nlm_values, n 

def smooth_coeff_matrix_biased(S, T, SS, TT, mass, nmax, lmax, mmax, sn):
    S_matrix_smooth = np.zeros((nmax+1, lmax+1, lmax+1))
    T_matrix_smooth = np.zeros((nmax+1, lmax+1, lmax+1))
    n_coeff_s = 0
    n_coeff_t = 0
    for n in range(nmax+1):
        for l in range(lmax+1):
            for m in range(l+1):
                bS_temp, ns = smoothing_biased(SS[n][l][m], S[n][l][m], mass, sn)
                bT_temp, nt = smoothing_biased(TT[n][l][m], T[n][l][m], mass, sn)
                S_matrix_smooth[n][l][m] = bS_temp*S[n][l][m]
                T_matrix_smooth[n][l][m] = bT_temp*T[n][l][m]
                n_coeff_s += ns                
                n_coeff_t += nt
                
    return S_matrix_smooth, T_matrix_smooth, n_coeff_s, n_coeff_t
    

#if __name__ == "__main__":
    
def coeff_uncorrelated(S, T, SS, TT, ST, mass, sn=0, verb=False):
    cov_matrix = covariance_matrix_builder(S, T, SS, TT, ST, mass)
    # SVD decomposition of the covariance matrix
    T_rot, v, TL = linalg.svd(cov_matrix)
    
    # Computes inverted transformation matrix
    T_rot_inv = linalg.inv(T_rot)

    # Variances of the coefficients in the uncorrelated base.
    varS = v[0]
    varT = v[1]
    
    ## uncorrelated coefficients
    coeff_base = np.array([S, T])
    S_unc, T_unc = np.dot(T_rot, coeff_base)
    b_S_unc, b_T_unc = smoothing(S_unc, T_unc, varS, varT)

         
            
    return S_unc, varS, b_S_unc
 

    

