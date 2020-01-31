#!/usr/bin/env python3.8
"""
Pipeline to generate ascii files of the MW particles, LMC bound particles and
MW+LMC unbound particles

author: github/jngaravitoc
12/2019


    Code Features:
        - Compute BFE expansion from a collection of snapshots
        - It separates a satellite galaxy from a host galaxy 
        - Compute COM of satellite and host galaxy
        - Compute bound and satellite unbound particles
        - Run in parallel for the nlm list! 
    
    TODO:

        Parameter file:
            - Make different categories for Host and Satellite?
            - Think that if this is going to be general we might need more than
              one satellite. 

        Implement all optional outputs:
            - random satellite sample
            - output ascii files
            - what if the COM is provided?
            - use ids to track bound - unbound particles -- think about cosmo
              zooms
            - track bound mass fraction
            
        Implement checks:
            - equal mass particles (DONE)
            - com accuracy check
            - plot figure with com of satellite and host for every snapshot..
            - BFE monopole term amplitude -- compute nmax=20, lmax=0 and check
                larger term is 000

        Implement tests for every function**
        Implement parallel computation for bound satellite particles.
        * : fast to implement
        ** : may need some time to implement

        Paralleization:
            - Why results=pool() returns empty list if --ncores==1?

    - known issues:
        - currently multiprocessing return the following error when many
          particles are used: 
          struct.error: 'i' format requires -2147483648 <= number <= 2147483647

          This is a known issue of multiprocessing that apparently is solved in
          python3.8 
          see :
            https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
"""

import numpy as np
import sys
import schwimmbad
import LMC_bounded as lmcb
import gadget_to_ascii as g2a
import reading_snapshots as reads
import coeff_parallel as cop
import allvars
from argparse import ArgumentParser
from quick_viz_check import scatter_plot

class BFECoeff:
    def __init__(self, pos, mass):
        self.pos = pos
        self.mass = mass

    def main(self, pool, nmax, lmax, r_s, var=True):
        worker = cop.Coeff_parallel(self.pos, self.mass, r_s, var)
        tasks = cop.nlm_list(nmax, lmax)
        results = pool.map(worker, tasks)
        pool.close()
        return results


#def main(pool, nmax, lmax, r_s, var=True):
#    worker = cop.Coeff_parallel(pos, mass, r_s, var)
#    tasks = cop.nlm_list(nmax, lmax)
#    results = pool.map(worker, tasks)
#    pool.close()
#    return results



if __name__ == "__main__":

    #snap_names = ["MWLMC3_100M_new_b0_090",
    #			  "MWLMC3_100M_new_b1_091",
    #			  "MWLMC4_100M_new_b0_114",
    #             "MWLMC4_100M_new_b1_115",
    #			  "MWLMC5_100M_new_b1_110",
    #			  "MWLMC5_100M_new_b0_109",
    #			  "MWLMC6_100M_new_b0_2_113",
    #			  "MWLMC6_100M_new_b1_2_114"]


    parser = ArgumentParser(description="Parameters file for bfe-py")

    parser.add_argument("--param", dest="paramFile", default="config.yaml",
                       type=str, help="provide parameter file")


    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=16,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    global args
    args = parser.parse_args()
    

    paramfile = args.paramFile
    params = allvars.readparams(paramfile)
    in_path = params[0]
    snapname = params[1]
    outpath = params[2]
    out_name = params[3]
    n_halo_part = params[4]
    npart_sample = params[5]
    nmax = params[6]
    lmax = params[7]
    rs = params[8]
    ncores = params[9]
    mpi = params[10]
    rcut_halo = params[11]
    init_snap=params[12]
    final_snap=params[13]
    SatBFE = params[14] 
    #for i in range(0, len(snap_names)):


    for i in range(init_snap, final_snap):
        print("**************************")
        # Loading data:
        halo = reads.read_snap_coordinates(in_path, snapname+"_{:03d}".format(i), n_halo_part, com_frame='MW', galaxy='MW')
        # Truncates halo:
        if rcut_halo>0:
            print("Truncating halo particles at {} kpc".format(rcut_halo))
            pos_halo_tr, vel_halo_tr, mass_tr, ids_tr = g2a.truncate_halo(halo[0], halo[1], halo[3], halo[4], rcut_halo)
            del halo
        else : 
            pos_halo_tr = halo[0]
            vel_halo_tr = halo[1]
            mass_tr = halo[3]
            ids_tr = halo[4]

        if npart_sample>0: 
            print("Sampling halo particles with: {} particles".format(npart_sample))
            pos_halo_tr, vel_halo_tr, mass_tr = g2a.sample_halo(pos_halo_tr, vel_halo_tr, mass_tr, npart_sample) 

        if SatBFE == 1:
            satellite = reads.read_snap_coordinates(in_path, snapname+"_{:03d}".format(i), n_halo_part, com_frame='sat', galaxy='sat')
            pos_sat_tr, vel_sat_tr, mass_sat_tr, ids_sat_tr = g2a.truncate_halo(satellite[0], satellite[1], satellite[3], satellite[4], rcut_halo)
            pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em = g2a.npart_satellite(pos_sat_tr, vel_sat_tr, ids_sat_tr, mass_sat_tr[0], mass_tr[0])
        print(len(pos_halo_tr))
        scatter_plot(outpath+snapname+"_{:03d}".format(i), pos_halo_tr)
        """
        assert np.abs(mass_sat_em[0]/mass_tr[0]-1)<1E-3, 'Error: particle mass of satellite different to particle mass of the halo'
        
        
        # Outs: 
        if snaps_ascii==True:
            out_snap_host = 'MW_{}_{}'.format(int(len(pos_halo_tr)/1E6), snapname+"{}".format(i))
            out_snap_sat= 'LMC_{}_{}'.format(int(len(pos_sat_em)/1E6), snapname+"{}".format(i))
            #write_log([n_halo_part, halo[3][0], len(pos_sample), mass_sample], [len(pos_sat_tr[0]), satellite[3][0], len(pos_sat_em), mass_sat_em])
            write_snap_txt(out_path_MW, out_snap_host, pos_halo_tr, vel_halo_tr, mass_tr, ids_tr)
            write_snap_txt(out_path_LMC, out_snap_sat, pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em)
            #write_snap_txt(out_path_LMC, out_snap_sat, satellite[0], satellite[1], satellite[3], satellite[4])
    
        # Satellite bound particles
        #pos, vel, mass, ids = reading_particles(snapname)
        # TODO: parallelize this part! 
        print('Computing bound particles!')
        armadillo = lmcb.find_bound_particles(pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em, args.rs, args.nmax, args.lmax)
       
        print('Bound particles computed')

        pos_bound = armadillo[0]
        vel_bound = armadillo[1]
        N_bound =  armadillo[2]
        ids_bound =  armadillo[3]
        pos_unbound = armadillo[4]
        vel_unbound = armadillo[5]
        ids_unbound = armadillo[6]

        lmc_bound = np.array([pos_bound[:,0], pos_bound[:,1], pos_bound[:,2],
                              vel_bound[:,0], vel_bound[:,1], vel_bound[:,2],
                              ids_bound]).T

        lmc_unbound = np.array([pos_unbound[:,0], pos_unbound[:,1], pos_unbound[:,2],
                                vel_unbound[:,0], vel_unbound[:,1], vel_unbound[:,2],
                                ids_unbound]).T
        print('Combining satellite unbound particles with host particles')
        	
        
        pos_host_sat = np.vstack((pos_halo_tr, pos_unbound))		
        vel_host_sat = np.vstack((vel_halo_tr, vel_unbound))
        #ids_host_sat = np.hstack((ids_halo_tr, ids_unbound))
        #fbound_mass = len(pos_sat)
        mass_array = np.ones(len(pos_host_sat[:,0]))*mass_sat_em[0]
        print(mass_array[0])
        mw_lmc_unbound = np.array([pos_host_sat[:,0], pos_host_sat[:,1], pos_host_sat[:,2], 
                                   vel_host_sat[:,0], vel_host_sat[:,1], vel_host_sat[:,2],
                                   mass_array]).T

        ## Run bfe here! 
        ## TODO: quick test run BFE with lmax=0 and nmax=20 to check that the first term is the largest
        """

        # TODO: Change this parmetetrs file to the ones read from  config.yaml
        #armadillo = lmcb.find_bound_particles(pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em, 10, 20, 20)
        
        pool = schwimmbad.choose_pool(mpi=args.mpi,
                                      processes=args.n_cores)
        pos = np.copy(pos_halo_tr)
        mass = np.copy(mass_tr)
        print(mass[0])
        halo_coeff = BFECoeff(pos, mass)
        results = halo_coeff.main(pool, nmax, lmax, rs, var=True)
        results_r = np.array(results)
        #pos = pos_halo_tr
        #mass = mass_tr
        #results = main(pool, nmax, lmax, rs, var=True)
        print('Done computing coefficients')
        Snlm=results_r[:,0]
        Tnlm=results_r[:,1]
        varSnlm=results_r[:,2]
        varTnlm=results_r[:,3]
        varSTnlm=results_r[:,4]

        #halo_pot = BFEPot(Snlm, Tnlm, rs)
        #results_pot = halo_pot.main(pool, pos)

        cop.write_coefficients(outpath+out_name+"snap_{:0>3d}.txt".format(i), Snlm, varSnlm, Tnlm,  varTnlm, varSTnlm, nmax, lmax, rs, mass[0])
        #print('Done writing coefficients')
        
