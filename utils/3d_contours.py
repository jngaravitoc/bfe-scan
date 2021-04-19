#import biff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mayavi import mlab
from mayavi.api import Engine
import cmasher as cmr

font = {'size':18, 'family':'serif'}
matplotlib.rc('font', **font)


#Function that reads the N-body simulation orbit
def reading_Nbody(snap_name):
    data = np.loadtxt(snap_name)
    #time = data[:,0]
    #Rgal = data[:,1]
    x_sat= data[:,6]
    y_sat = data[:,7]
    z_sat = data[:,8]
    x_gal = data[:,0]
    y_gal = data[:,1]
    z_gal = data[:,2]
    #Vgal = data[:,8]
    vx_sat = data[:,9]
    vy_sat = data[:,10]
    vz_sat = data[:,11]
    vx_gal = data[:,3]
    vy_gal = data[:,4]
    vz_gal = data[:,5]
    Rgal= np.sqrt((x_sat-x_gal)**2 + (y_sat-y_gal)**2 + (z_sat-z_gal)**2)
    Vgal= np.sqrt((vx_sat-vx_gal)**2 + (vy_sat-vy_gal)**2 + (vz_sat-vz_gal)**2)

    return Rgal, x_sat, y_sat, z_sat, x_gal, y_gal, z_gal, Vgal, vx_sat, vy_sat, vz_sat, vx_gal, vy_gal, vz_gal

def load_density(dens_fname, nbins):
    rho_mwlmc = np.loadtxt(dens_fname)
    rho_matrix = np.reshape(rho_mwlmc[:,0], (nbins, nbins, nbins))
    return rho_matrix, rho_mwlmc[:,1], rho_mwlmc[:,2], rho_mwlmc[:,3]

def mayavi_plot(xlmc, ylmc, zlmc, rho, angle, figname, snap, i, distance=1500):
    engine = Engine()
    engine.start()
    fig = mlab.figure(size=(227*2, 170*2), bgcolor=(0, 0, 0))



    #rho1 = mlab.pipeline.add_dataset(rho)
    #iso = mlab.contour3d(rho, opacity=0.99,
    #                    extent=[-400, 400, -400, 400, -400, 400],
    #                    vmin=-0.23, vmax=0.225, transparent=True)
    # vmax = 0.225

    #scene = engine.scenes[0]
    #iso_surface = scene.children[0].children[0].children[0]


    #iso_surface.compute_normals = False  # without this only 1 contour will be displayed

    # LMC values
    #iso_surface.contour.number_of_contours = 3
    #iso_surface.contour.minimum_contour = -4.5
    #iso_surface.contour.maximum_contour = -3.6

    # MW halo values
    #iso_surface.contour.number_of_contours = 11
    #iso_surface.contour.minimum_contour = -7
    #iso_surface.contour.maximum_contour = -2.9

    # DM wake values
    #iso_surface.contour.minimum_contour = -0.23
    #iso_surface.contour.maximum_contour = 0.223
    #mlab.colorbar()

    # DM Debris values
    #so_surface.contour.number_of_contours = 5
    #so_surface.contour.minimum_contour = 0.19
    #iso_surface.contour.maximum_contour = 0.5
    #mlab.colorbar()

    ## trasnparency & colormap
    x = np.linspace(0, 1, 256)
    colors = plt.cm.get_cmap(plt.get_cmap(i))(x)*255

    #rgb = cmr.take_cmap_colors(i, N=256)
    #rgb_array = np.array([rgb])*255
    #colors = np.ones((256, 4))
    #colors[:,:3] = rgb_array
    #iso.module_manager.scalar_lut_manager.lut.table = colors
    #lut = iso_surface.module_manager.scalar_lut_manager.lut.table.to_array()
    #lut[:, -1] = np.linspace(0, 255, 256)
    #lut[:, -1] = np.logspace(0, np.log10(255), 256)
    #lut[:, -1] = np.linspace(0, np.sqrt(255), 256)**2
    #lut[:,-1] = np.ones(256)*100
    #iso_surface.module_manager.scalar_lut_manager.lut.table = lut
    #mlab.draw()
    # the following line will print you everything that you can modify on that object
    #iso_surface.contour.print_traits()

    #mlab.plot3d(xlmc[0:snap+1], ylmc[0:snap+1], zlmc[0:snap+1], color=(1, 1, 1), opacity=0.9, line_width=20, tube_radius=1)
    mlab.points3d(xlmc[snap], ylmc[snap], zlmc[snap], scale_factor=5,
                color=(1, 1, 1), opacity=1, resolution=100)


    #mlab.plot3d(xlmc[0:snap+1], ylmc[0:snap+1], zlmc[0:snap+1], color=(215/255,  27/255, 96/255), opacity=0.9, line_width=20, tube_radius=1)
    #mlab.points3d(xlmc[snap], ylmc[snap], zlmc[snap], scale_factor=5,
    #             color=(215/255, 27/255, 96/255), opacity=1, resolution=100)
    mlab.points3d(0, 0, 0, scale_factor=20, color=(30/255,136/255,229/255), opacity=1,
                 resolution=300, mode='2dcircle', line_width=8)
    #inner shell of observations
    mlab.points3d(0, 0, 0, scale_factor=100, color=(0, 0, 0), opacity=0.6,  resolution=300)


    #mlab.points3d(0, 50, 0, scale_factor=10, color=(215/255, 27/255, 96/255), opacity=1,
    #              resolution=300)

    # outter shell of observations
    mlab.points3d(0, 0, 0, scale_factor=200, color=(255/255,210/255,78/255), opacity=0.3,
                     resolution=300)

    #lut =tt.module_manager.scalar_lut_manager.lut.table.to_array()
    #ilut = lut[::-1]
    ## putting LUT back in the surface object
    #tt.module_manager.scalar_lut_manager.lut.table = ilut
    #tt.actor.actor.rotate_z(270)
    #mlab.roll(270)
    #limits




    mlab.view(azimuth=angle, distance=distance, elevation=100.0)

    #z3 = mlab.plot3d(xlmc[:snap], ylmc[:snap], zlmc[:snap], color=(1,0,0), opacity=1)
    #z4 = mlab.points3d(xlmc[snap], ylmc[snap], zlmc[snap], scale_factor=20,  color=(1,0,0), opacity=1, resolution=100)
    #z3 = mlab.points3d(0, 0, 0, scale_factor=900, color=(0,0,0), opacity=1,
    #                  resolution=100, mode='2dcircle', line_width=10)
    #z4 = mlab.text3d(0, -500, 250, '100 kpc', scale=30, color=(0,0,0))
    #z5 = mlab.plot3d([0, 0], [-500, -400], [240,240], line_width=15, color=(0,0,0))

    #iso_3d_plot = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
    mlab.savefig("./"+i+"_black_bg_distance_"+str(distance)+figname, size=(227*2, 170*2))
    mlab.close()
    engine.stop()

    return 0#iso_3d_plot

def shell_cut(rho, xgrid, ygrid, zgrid, nbins, rmin, rmax):
    r_g = np.sqrt(xgrid**2+ygrid**2+zgrid**2)
    rcut = np.where((r_g>rmax) | (r_g<rmin))
    rho_flat = np.copy(rho.flatten())
    rho_flat[rcut]= -0
    rho_cut_3d = np.reshape(rho_flat, (nbins, nbins, nbins))
    return rho_cut_3d



def zoom_effect(xlmc, ylmc, zlmc, rho, angle, figname, snap, color, dmin, dmax, delta_d, zoom='out'):
    d_camara = np.arange(dmin, dmax, delta_d)
    if zoom == "out":
        for d in range(len(d_camara)):
            mayavi_plot(xlmc, ylmc, zlmc, rho, angle, figname, snap=snap, i=color, distance=d_camara[i])
    elif zoom == "in":
        for d in range(len(d_camara)):
            mayavi_plot(xlmc, ylmc, zlmc, rho, angle, figname, snap=snap, i=color, distance=d_camara[::-i])

#def rotation_effect()

def iso_dens_plot(threed_render, figname):
    print('here')
    plt.imshow(threed_render)
    plt.savefig(figname, bbox_inches='tight')

if __name__ == "__main__":

### lmc orbit

    LMC5_b1 = '../../MW_anisotropy/data/orbits/LMC5_100Mb1_orbit.txt'
    LMC5_b1_orbit = reading_Nbody(LMC5_b1)

    x_sat = LMC5_b1_orbit[1]
    y_sat = LMC5_b1_orbit[2]
    z_sat = LMC5_b1_orbit[3]
    x_gal = LMC5_b1_orbit[4]
    y_gal = LMC5_b1_orbit[5]
    z_gal = LMC5_b1_orbit[6]
    #bins_phys = 400/200.

    xlmc = (x_sat[:112]-x_gal[:112])+5#*bins_phys + 2.5* bins_phys
    ylmc = (y_sat[:112]-y_gal[:112])+5#*bins_phys + 2.5* bins_phys
    zlmc = (z_sat[:112]-z_gal[:112])+5#-1.2#*bins_phys + 2.5* bins_phys
    print(xlmc[110], ylmc[110], zlmc[110])
    nbins = 300
    #colors = ['afmhot', 'gist_heat', 'inferno', 'magma', 'YlGnBu']
    #color=['afmhot']
    #colors2 = ['cmr.amber', 'cmr.arctic', 'cmr.freeze', 'cmr.voltage']
    colors2 = ['cmr.freeze']
    #colors2 = ['cmr.amber']
    #densities = 'rho_mwlmc_bfe.txt'
    #densities1 = 'rhomwlmc.txt'
    #densities2 = 'rhomwlmc_base.txt'
    for i in range(109, 110, 1):
        #densities1 = './data/rho_wake_debris5_bfe_b1_120_r_{:0>3d}.txt'.format(i)
        #densities1 = './data/rho_lmc_bfe_b1_120_r_{:0>3d}.txt'.format(i)
        #densities1 = './data/rho_mwlmc_bfe_120_r_{:0>3d}.txt'.format(i)
        ## wake data:
        #densities1 = './data/rho_wake_bfe_120_r_{:0>3d}.txt'.format(i)
        #densities1 = '../bfe/analysis/test_rho_mwlmc5_bfe_120_r_{:0>3d}.txt'.format(i)
        #densities1 = "../bfe/analysis/test_600_rho_mwlmc5_1_monopole_bfe_300_r_{:0>3d}.txt".format(i)
        #densities1 = "../bfe/analysis/test_400_rho_mwlmc5_0_monopole_bfe_200_r_{:0>3d}.txt".format(i)

        #densities1 = './data/rho_mw_bfe_120_r_{:0>3d}.txt'.format(i)
        #rho1, xg, yg, zg = load_density(densities1, nbins)
        #rho_shell = shell_cut(rho1, xg, yg, zg, nbins=nbins, rmin=10, rmax=300)
        #ho2 = load_density(densities2, nbins)
        for j in range(-1, 2, 1):
            #figname = './plots/lmc_snap_{:0>3d}_angle_{:0>3d}.png'.format(i, j)
            #mayavi_plot(xlmc, ylmc, zlmc, np.log10(np.abs(rho1)), 0, figname, snap=i)
            for d in range(1200, 1205, 5):
                figname = 'mw_lmc_shells_{:0>3d}_120_b1_angle_{:0>3d}_distance_{:0>3d}.png'.format(i, j, d)
                mayavi_plot(xlmc, ylmc, zlmc, 1, j, figname, snap=i, i=colors2[0], distance=d)
            #iso_dens_plot(iso_contours, figname)

#for i in range(0, 360):
#    mlab.view(azimuth=i, elevation=90, distance=1200)
