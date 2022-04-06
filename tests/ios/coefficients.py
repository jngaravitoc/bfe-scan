import numpy as np
from bfe.ios import write_coefficients_hdf5, read_coefficients


if __name__ == "__main__":
	#Load coefficients
	coefficients = np.loadtxt("../BFE_MW2_100M_grav_MO3_COM_n20_20_host_snap_000.txt")
	nmax=10
	lmax=2
	mmax=1
	r_s=40.85
	pm = 1.3E-4
	rcom = [-0.5, -0.4, -0.4]
	G = 1
	file_name = "coeffecients_io_test.hdf5"
	# dt, snap name
	S = coefficients[:,0].reshape(11, 3, 2)
	T = coefficients[:,1].reshape(11, 3, 2)
	# Write coefficients
	write_coefficients_hdf5(file_name, [S, T], nmax, lmax, mmax, r_s, pm, rcom, G)
	# Load hdf5 coefficients
	coef, exp_length, exp_var, rcom = read_coefficients(file_name)
	print(exp_length, exp_var, rcom)
	print(exp_length[0])
