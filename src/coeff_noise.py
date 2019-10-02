import numpy as np


## Reading coefficients
def reshape_matrix(matrix, n, l, m):
    col_matrix = np.zeros((n+1, l+1, m+1))
    counter = 0
    for n in range(n+1):
        for l in range(l+1):
            for m in range(0, l+1):
                col_matrix[n][l][m] = matrix[counter]
                counter+=1
    return col_matrix

## Reading coefficients
def read_coeff_matrix(filename, nfiles, n, l, m, nmin=0, nmax=1000):
    print(nmax)

    S_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    T_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    
    S_mean = np.zeros((int((n+1)*(l+1)*(l/2.+1))))
    T_mean = np.zeros((int((n+1)*(l+1)*(l/2.+1))))

    for i in range(nmin, nmax):
        coeff = np.loadtxt(filename + '{:03d}.txt'.format(i))
        S_matrix[:,i] = coeff[:,0]
        T_matrix[:,i] = coeff[:,1]
        
    for i in range(len(S_matrix[:,0])):
        S_mean[i] = np.mean(S_matrix[i])
        T_mean[i] = np.mean(T_matrix[i])

    S_mean_matrix = reshape_matrix(S_mean, n, l, m)
    T_mean_matrix = reshape_matrix(T_mean, n, l, m)

    return S_mean_matrix, T_mean_matrix


## Reading covariance
def var_matrix(filename, filename2, nfiles, n, l, m, mass, nmin=0, nmax=1000):
    print(nmax)

    Scov_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    Tcov_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    
    Svar_mean = np.zeros((int((n+1)*(l+1)*(l/2.+1))))
    Tvar_mean = np.zeros((int((n+1)*(l+1)*(l/2.+1))))
    
    
    S_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    T_matrix = np.zeros((int((n+1)*(l+1)*(l/2.+1)), nfiles))
    
    for i in range(nmin, nmax):
        cov = np.loadtxt(filename + '{:03d}.txt'.format(i))
        coeff = np.loadtxt(filename2 + '{:03d}.txt'.format(i))

        Scov_matrix[:,i] = cov[:,0]
        Tcov_matrix[:,i] = cov[:,1]
        
        S_matrix[:,i] = coeff[:,0]
        T_matrix[:,i] = coeff[:,1]
    

    for i in range(len(Scov_matrix[:,0])):
        Svar_mean[i] = np.mean((Scov_matrix[i] - mass*S_matrix[i]**2)) #/ nfiles
        Tvar_mean[i] = np.mean((Tcov_matrix[i] - mass*T_matrix[1]**2)) #/ nfiles

    Svar_mean_matrix = reshape_matrix(Svar_mean, n, l, m)
    Tvar_mean_matrix = reshape_matrix(Tvar_mean, n, l, m)
        
    return Svar_mean_matrix, Tvar_mean_matrix


def copy_matrix(M, indices):
    M_new = np.zeros(shape(M))
    for i in range(len(indices[0])):
        M_new[indices[0][i]][indices[1][i]][indices[2][i]] = M[indices[0][i]][indices[1][i]][indices[2][i]]
    return M_new

def smoothing(var_coeff, coeff):
    """
    Coefficients smoothing 
    
    """
    b_nlm = 1/(1 + var_coeff/coeff**2)
    # This line remove nans an put the original values
    b_nlm_values = np.nan_to_num(b_nlm)
    return b_nlm_values

def coefficients_smooth_level(S, T, bs, bt, b_cut, verb=0):
    """
    Returns coefficients with energy higher than e_cut
    e_cut : float 
        between 0 and 1
        
    """
 
    bs_cut_index = np.where(bs>b_cut)
    bt_cut_index = np.where(bt>b_cut)
    
    if verb == 1:
        print('N coeff S= ', len(bs_cut_index[0]))
        print('N coeff T= ', len(bt_cut_index[0]))

    S_new = copy_matrix(S, bs_cut_index)
    T_new = copy_matrix(T, bt_cut_index)
    
    return S_new, T_new, len(bs_cut_index[0]), len(bt_cut_index[0])
    
    
    


