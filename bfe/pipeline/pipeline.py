#!/usr/bin/env python3.8
"""
bfe-py 
is a python code that computes BFE expansion in idealized n-body simulations
it works in parallel using multiprocessing.

Pipeline to generate ascii files of the MW particles, LMC bound particles and
MW+LMC unbound particles

author: github/jngaravitoc
12/2019

"""

import numpy as np
import sys
import schwimmbad
import bfe.satellites as lmcb
import bfe.ios.gadget_to_ascii as g2a
import bfe.ios.io_snaps as ios
import bfe.coefficients.parallel_coefficients as cop
import allvars

from argparse import ArgumentParser
from quick_viz_check import scatter_plot


if __name__ == "__main__":

    parser = ArgumentParser(description="Parameters file for bfe-py")

    parser.add_argument(
            "--param", dest="paramFile", default="config.yaml",
            type=str, help="provide parameter file")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
            "--ncores", dest="n_cores", default=16,
            type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    global args
    args = parser.parse_args()

    # TODO: move all these variables to allvars and call it from there
    # Loading paramfile
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
    init_snap = params[12]
    final_snap = params[13]
    SatBFE = params[14]
    sat_rs = params[15]
    nmax_sat = params[16]
    lmax_sat = params[17]
    HostBFE = params[18]
    SatBoundParticles = params[19]
    HostSatUnboundBFE = params[20]
    write_snaps_ascii = params[21]
    out_ids_bound_unbound_sat = params[22]
    plot_scatter_sample = params[23]
    npart_sample_satellite = params[24]
    # rcut_sat = params[26]
   


    for i in range(init_snap, final_snap):
        with open(outpath+'info.log', 'a') as out_log:
            out_log.write("**************************\n")
            out_log.write("loading snap {}{} \n".format(snapname, i))

            # *********************** Loading data: **************************
            if ((HostBFE == 1) | (HostSatUnboundBFE == 1)):
                out_log.write("reading host particles")
                halo = ios.read_snap_coordinates(
                        in_path, snapname+"_{:03d}".format(i),
                        n_halo_part, com_frame='MW', galaxy='MW')
                rcom_halo = halo[5]
                vcom_halo = halo[6]
                # Truncates halo:
                if rcut_halo > 0:
                    out_log.write(
                            "Truncating halo particles at {} kpc \n".format(
                                rcut_halo))

                    pos_halo_tr, vel_halo_tr, mass_tr, ids_tr \
                            = g2a.truncate_halo(
                                    halo[0], halo[1],
                                    halo[3], halo[4], rcut_halo)
                    del halo
                else:
                    pos_halo_tr = halo[0]
                    vel_halo_tr = halo[1]
                    mass_tr = halo[3]
                    ids_tr = halo[4]
                    del halo
                # Sampling halo

                if npart_sample > 0:
                    out_log.write(
                            "Sampling halo particles with: {} particles \n".format(
                                npart_sample))

                    pos_halo_tr, vel_halo_tr, mass_tr, ids_tr \
                            = g2a.sample_halo(
                                    pos_halo_tr, vel_halo_tr,
                                    mass_tr, npart_sample, ids_tr)

            # Truncating satellite for BFE computation
            if ((SatBFE == 1) | (HostSatUnboundBFE == 1)):
                out_log.write("reading satellite particles \n")
                
                satellite = ios.read_snap_coordinates(
                        in_path, snapname+"_{:03d}".format(i),
                        n_halo_part, com_frame='sat', galaxy='sat')

                pos_sat_tr, vel_sat_tr, mass_sat_tr, ids_sat_tr\
                        = g2a.truncate_halo(
                                satellite[0], satellite[1], satellite[3],
                                satellite[4], rcut_halo)

                if ((HostBFE == 1) | (HostSatUnboundBFE == 1)):

                    pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em\
                            = g2a.npart_satellite(
                                    pos_sat_tr, vel_sat_tr, ids_sat_tr,
                                    mass_sat_tr[0], mass_tr[0])

                    assert np.abs(mass_sat_em[0]/mass_tr[0]-1) < 1E-3,\
                            'Error: particle mass of satellite different to particle mass of the halo'

                if npart_sample_satellite > 0:
                    pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em\
                            = g2a.sample_halo(
                                    pos_sat_tr, vel_sat_tr, mass_sat_tr,
                                    npart_sample_satellite, ids_sat_tr)

                else :
                    pos_sat_em = pos_sat_tr
                    vel_sat_em = vel_sat_tr
                    mass_sat_em = mass_sat_tr
                    ids_sat_em = ids_sat_tr
                    
                    del(pos_sat_tr)
                    del(vel_sat_tr)
                    del(mass_sat_tr)
                    del(ids_sat_tr)

            if plot_scatter_sample == 1:
                # Plot 2d projections scatter plots

                if ((HostBFE == 1) | (HostSatUnboundBFE == 1)):
                    scatter_plot(
                            outpath+snapname+"_host_{:03d}".format(i),
                            pos_halo_tr)

                if SatBFE == 1:
                    scatter_plot(
                            outpath+snapname+"_sat_{:03d}".format(i), 
                            pos_sat_em)
                
                    
            #*************************  Compute BFE: ***************************** 
    
            if ((SatBFE == 1) & (SatBoundParticles == 1)):
                out_log.write("Computing satellite bound particles!\n")

                armadillo = lmcb.find_bound_particles(
                        pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em, 
                        sat_rs, nmax_sat, lmax_sat, ncores,
                        npart_sample = 100000)

                # npart_sample sets the number of particles to compute the
                # potential in each cpu more than 100000 usually generate memory
                # errors

                # removing old variables
                del(pos_sat_em)
                del(vel_sat_em)
                

                out_log.write('Done: Computing satellite bound particles! \n')
                pos_bound = armadillo[0]
                vel_bound = armadillo[1]
                ids_bound = armadillo[2]
                pos_unbound = armadillo[3]
                vel_unbound = armadillo[4]
                ids_unbound = armadillo[5]
                rs_opt = armadillo[6]
                # mass arrays of bound and unbound particles
                N_part_bound = len(ids_bound)
                N_part_unbound = len(ids_unbound)
                mass_bound_array = np.ones(N_part_bound)*mass_sat_em[0]
                mass_unbound_array = np.ones(N_part_unbound)*mass_sat_em[0]
                
                # Mass bound fractions
                Mass_bound = (N_part_bound/len(ids_sat_em))*np.sum(mass_sat_em)
                Mass_unbound = (N_part_unbound/len(ids_sat_em))*np.sum(mass_sat_em)
                Mass_fraction = (N_part_bound)/len(ids_sat_em)
                out_log.write("Satellite bound mass fraction {} \n".format(Mass_fraction))
                out_log.write("Satellite bound mass {} \n".format(Mass_bound))
                out_log.write("Satellite unbound mass {} \n".format(Mass_unbound))

                if plot_scatter_sample == 1:
                    out_log.write("plotting scatter plots of unbound and bound satellite particles \n")
                    scatter_plot(outpath+snapname+"_unbound_sat_{:03d}".format(i), pos_unbound)
                    scatter_plot(outpath+snapname+"_bound_sat_{:03d}".format(i), pos_bound)
                
                if out_ids_bound_unbound_sat == 1:
                    out_log.write("writing satellite bound id \n")
                    np.savetxt(outpath+snapname+"_bound_sat_ids_{:03d}".format(i), ids_bound)

            if HostSatUnboundBFE == 1:
                pool_host_sat = schwimmbad.choose_pool(mpi=args.mpi,
                    processes=args.n_cores)
                out_log.write("Computing Host & satellite debris potential \n")
                # 'Combining satellite unbound particles with host particles')
                pos_host_sat = np.vstack((pos_halo_tr, pos_unbound))
                # TODO : Check mass array?
                mass_Host_Debris = np.hstack((mass_tr, mass_unbound_array))
                halo_debris_coeff = cop.Coeff_parallel(
                        pos_host_sat, mass_Host_Debris, rs, True, nmax, lmax)

                results_BFE_halo_debris = halo_debris_coeff.main(pool_host_sat)
                out_log.write("Done computing Host & satellite debris potential")
                ios.write_coefficients(
                        outpath+out_name+"_host_sat_unbound_snap_{:0>3d}.txt".format(i),
                        results_BFE_halo_debris, nmax, lmax, rs,
                        mass_Host_Debris[0], rcom_halo, vcom_halo)
            
    
            if HostBFE == 1:
                pool_host = schwimmbad.choose_pool(mpi=args.mpi,
                    processes=args.n_cores)
                out_log.write("Computing Host BFE \n")
                halo_coeff = cop.Coeff_parallel(
                        pos_halo_tr, mass_tr, rs, True, nmax, lmax)
                
                results_BFE_host = halo_coeff.main(pool_host)
                
                out_log.write("Done computing Host BFE")
                ios.write_coefficients(
                        outpath+out_name+"_host_snap_{:0>3d}.txt".format(i),
                        results_BFE_host, nmax, lmax, rs, mass_tr[0], rcom_halo, vcom_halo)
        

            if ((SatBFE == 1) & (SatBoundParticles == 1)):
                out_log.write("Computing Sat BFE \n")
                pool_sat = schwimmbad.choose_pool(mpi=args.mpi,
                                                  processes=args.n_cores)
                sat_coeff = cop.Coeff_parallel(
                        pos_bound, mass_bound_array, sat_rs, True, nmax_sat, lmax_sat)

                results_BFE_sat = sat_coeff.main(pool_sat)
                out_log.write("Done computing Sat BFE \n")
    
                ios.write_coefficients(
                        outpath+out_name+"_sat_bound_snap_{:0>3d}.txt".format(i),
                        results_BFE_sat, nmax_sat, lmax_sat, sat_rs,
                        mass_bound_array[0], satellite[5], satellite[6])
        
            if ((SatBFE == 1) & (SatBoundParticles == 0)):
                out_log.write("Computing Sat BFE \n")
                pool_sat = schwimmbad.choose_pool(mpi=args.mpi,
                                                  processes=args.n_cores)
                sat_coeff = cop.Coeff_parallel(
                        pos_sat_em, mass_sat_em, sat_rs, True, nmax_sat, lmax_sat)

                results_BFE_sat = sat_coeff.main(pool_sat)
                out_log.write("Done computing Sat BFE \n")
    
                ios.write_coefficients(
                        outpath+out_name+"_sat_snap_{:0>3d}.txt".format(i),
                        results_BFE_sat, nmax_sat, lmax_sat, sat_rs,
                        mass_sat_em[0], satellite[5], satellite[6])
        
            
    
            # TODO : check this flag 
            # Write snapshots ascii files
            if write_snaps_ascii == 1:
    
                # Write Host snap 
                if HostBFE == 1: 
                    out_snap_host = 'MW_{}_{}'.format(
                            int(len(pos_halo_tr)/1E6), snapname+"_{}".format(i))

                    g2a.write_snap_txt(
                            outpath, out_snap_host, pos_halo_tr,
                            vel_halo_tr, mass_tr, ids_tr)

                if SatBFE == 1:
                    out_snap_sat_bound= 'LMC_bound_{}'.format(
                            snapname+"_{}".format(i))

                    # Combining satellite unbound particles with host particles
                    g2a.write_snap_txt(
                        outpath, out_snap_sat_bound, pos_bound,
                        vel_bound, mass_bound_array, ids_bound)
