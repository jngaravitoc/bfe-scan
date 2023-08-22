import numpy as np



def power_matrices(power):
    """
    Compute the sum of the gravitational power of using the coefficients  over n, l and m. 
    """
    assert len(np.shape(power)) == 4, 'Power should be a matrix of shape (time, n, l, m)'

    tmax = np.shape(power)[0]
    nmax = np.shape(power)[1]
    lmax = np.shape(power)[2]
    mmax = np.shape(power)[3]
    
    print("Computing power for, tmax={}, nmax={}, lmax={}, mmax={}".format(tmax, nmax, lmax, mmax))
    # Compute power summing over l and m
    power_n = np.sum(np.sum(power, axis=2), axis=2)
    # Compute power summing over n, and m
    #print(np.shape(test_power))
    power_l = np.zeros((399, 21))
    power_lm = np.sum(power_host, axis=1)

    for l in range(lmax):
        power_l[:,l] = np.sum(power_lm[:,l,:l+1], axis=1)
        
    return power_n, power_l
