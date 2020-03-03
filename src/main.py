#!/usr/bin/env python3.8
"""
Pipeline to generate ascii files of the MW particles, LMC bound particles and
MW+LMC unbound particles

author: github/jngaravitoc
12/2019


    Code Features:
        - Compute BFE expansion from a collection of snapshots
        - Separates a satellite from its host by finding bound
          satellite particles.
        - Recenter Host and Satellite to its COM
        - Sample satellite particles to have the same mass of the host.
        - Run in parallel for the nlm list.
        - Write particle data in Gadget format if desired.

    TODO:

        Parameter file:
            - Make different categories for Host and Satellite?
            - Think that if this is going to be general we might need more than
            one satellite.

        Implement all optional outputs:
            - random satellite sample *
            - output ascii files
            - what if the COM is provided?
            - use ids to track bound - unbound particles -- think about cosmo
            zooms
            - track bound mass fraction
            - write Gadget format file
            - Check if all the flags are working
            - Flag: write_snaps_ascii
            - ids_tr is needed? or can be deleted to use less memory?

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

        Parallelization:
            - Fix results=pool() returns empty list if --ncores==1?

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
import io_snaps as ios
import coeff_parallel as cop
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
        with open('info.log', 'a') as out_log:
            out_log.write("**************************\n")
            out_log.write("loading snap {}{} \n".format(snapname, i))

            # *********************** Loading data: **************************
            if ((HostBFE == 1) | (HostSatUnboundBFE == 1)):
                halo = ios.read_snap_coordinates(
                        in_path, snapname+"_{:03d}".format(i),
                        n_halo_part, com_frame='MW', galaxy='MW')

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
                # TODO : return BFE of the bound particles the LMC
                satellite = ios.read_snap_coordinates(
                        in_path, snapname+"_{:03d}".format(i),
                        n_halo_part, com_frame='sat', galaxy='sat')

                pos_sat_tr, vel_sat_tr, mass_sat_tr, ids_sat_tr\
                        = g2a.truncate_halo(
                                satellite[0], satellite[1], satellite[3],
                                satellite[4], rcut_halo)

                # TODO : what if we are computing the BFE just for the satellite
                # particles
                print("Loading {} satellite particles".format(len(ids_sat_tr)))

                if ((HostBFE == 1) | (HostSatUnboundBFE == 1)):

                    pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em\
                            = g2a.npart_satellite(
                                    pos_sat_tr, vel_sat_tr, ids_sat_tr,
                                    mass_sat_tr[0], mass_tr[0])

                    assert np.abs(mass_sat_em[0]/mass_tr[0]-1) < 1E-3,\
                            'Error: particle mass of satellite different \\
                            to particle mass of the halo'
                elif npart_sample_satellite > 0:
                    pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em\
                            = g2a.sample_halo(
                                    pos_sat_tr, vel_sat_tr, mass_sat_tr,
                                    npart_sample_satellite, ids_sat_tr)
                else:
                    pos_sat_em = pos_sat_tr
                    vel_sat_em = vel_sat_tr
                    mass_sat_em = mass_sat_tr
                    ids_sat_em = ids_sat_tr
            # Plot 2d projections scatter plots
            if plot_scatter_sample == 1:

                if ((HostBFE == 1) | (HostSatUnboundBFE == 1)):
                    scatter_plot(
                            outpath+snapname+"_{:03d}".format(i),
                            pos_halo_tr)

                elif SatBFE == 1:
                    scatter_plot(
                            outpath+snapname+"_{:03d}".format(i), 
                            pos_sat_tr)
                
                    
            # *************************  Compute BFE: ***************************** 
    
            if ((SatBFE == 1) & (SatBoundParticles == 1)):
                print("Satellite total mass", np.sum(mass_sat_em))
                out_log.write("Computing satellite bound particles!\n")
                armadillo = lmcb.find_bound_particles(
                        pos_sat_em, vel_sat_em, mass_sat_em, ids_sat_em, 
                        sat_rs, nmax_sat, lmax_sat, ncores,
                        npart_sample = 100000)

                # npart_sample sets the number of particles to compute the
                # potential in each cpu more than 100000 usually generate memory
                # errors
                del(pos_sat_em)
                del(vel_sat_em)
                out_log.write('Done: Computing satellite bound particles!')
                pos_bound = armadillo[0]
                vel_bound = armadillo[1]
                #N_part_bound = armadillo[6]
                pot_bound = armadillo[6]
                ids_bound = armadillo[2]
                pos_unbound = armadillo[3]
                vel_unbound = armadillo[4]
                ids_unbound = armadillo[5]
                # Mass_bound = (len(ids_bound)/len(ids_sat_em))*np.sum(mass_sat_em)
                # Mass_unbound = (len(ids_unbound)/len(ids_sat_em))*np.sum(mass_sat_em)
                # Mass_fraction = (N_part_bound)/len(ids_sat_em)
                # print("Satellite bound mass fraction", Mass_fraction)
                # print("Satellite bound mass", Mass_bound)
                # print("Satellite unbound mass", Mass_unbound)
          
            # Pool for parallel computation! 
            pool = schwimmbad.choose_pool(mpi=args.mpi,
                                          processes=args.n_cores)

            if HostSatUnboundBFE == 1:
                out_log.write("Computing Host & satellite debris potential \n")
                # 'Combining satellite unbound particles with host particles')
                pos_host_sat = np.vstack((pos_halo_tr, pos_unbound))
                # TODO : Check mass array?
                mass_array = np.ones(len(ids_unbound))*mass_sat_em[0]
                mass_Host_Debris = np.hstack((mass_tr, mass_array))
                halo_debris_coeff = cop.Coeff_parallel(
                        pos_host_sat, mass_Host_Debris, rs, True, nmax, lmax)

                results_BFE_halo_debris = halo_debris_coeff.main(pool)
                out_log.write("Done computing Host & satellite debris potential")
                ios.write_coefficients(
                        outpath+out_name+"HostSatUnbound_snap_{:0>3d}.txt".format(i),
                        results_BFE_halo_debris, nmax, lmax, rs,
                        mass_Host_Debris[0])
            
    
            if HostBFE == 1:
                out_log.write("Computing Host BFE \n")
                halo_coeff = cop.Coeff_parallel(
                        pos_halo_tr, mass_tr, rs, True, nmax, lmax)
                results_BFE_host = halo_coeff.main(pool)
                out_log.write("Done computing Host BFE")
                ios.write_coefficients(
                        outpath+out_name+"Host_snap_{:0>3d}.txt".format(i),
                        results_BFE_host, nmax, lmax, rs, mass_tr[0])
        

            elif SatBFE == 1:
                out_log.write("Computing Sat BFE \n")
                mass_array = np.ones(len(ids_bound))*mass_sat_em[0]
                sat_coeff = cop.Coeff_parallel(
                        pos_bound, mass_array, sat_rs, True, nmax_sat, lmax_sat)

                results_BFE_sat = sat_coeff.main(pool)
                out_log.write("Done computing Sat BFE \n")
    
                ios.write_coefficients(
                        outpath+out_name+"Sat_snap_{:0>3d}.txt".format(i),
                        results_BFE_sat, nmax_sat, lmax_sat, rs, mass_array[0])
        
            
    
            # TODO : check this flag 
            # Write snapshots ascii files
            if write_snaps_ascii == 1:
    
                # Write Host snap 
                if HostBFE == 1: 
                    out_snap_host = 'MW_{}_{}'.format(
                            int(len(pos_halo_tr)/1E6), snapname+"{}".format(i))

                    g2a.write_snap_txt(
                            outpath, out_snap_host, pos_halo_tr,
                            vel_halo_tr, mass_tr, ids_tr)

                if SatBFE == 1:
                # write_snap_txt(out_path_LMC, out_snap_sat,
                # satellite[0], satellite[1], satellite[3], satellite[4])
                    out_snap_sat_bound= 'LMC_bound_{}'.format(
                            snapname+"{}".format(i))
                    # out_snap_sat_unbound= 'LMC_unbound_{}'.format(
                    # snapname+"{}".format(i))
                    # mass_bound = np.ones(N_part_bound)*mass_sat_em[0]
                    mass_unbound = np.ones(len(ids_unbound))*mass_sat_em[0]

                    # Combining satellite unbound particles with host particles
                    # g2a.write_snap_txt(outpath, out_snap_sat_bound, pos_bound,
                    # vel_bound, mass_bound, ids_bound)

                    g2a.write_snap_txt(
                            outpath, out_snap_sat_bound, pos_bound,
                            vel_bound, pot_bound, ids_bound)
