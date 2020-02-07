import biff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mayavi import mlab
from mayavi.api import Engine

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
    rho_matrix = np.reshape(rho_mwlmc, (nbins, nbins, nbins))
    return rho_matrix

def mayavi_plot(xlmc, ylmc, zlmc, rho, angle, figname):
    engine = Engine()
    engine.start()
    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0))

    #mlab.plot3d(, x_sat[:111]-x_gal[:111]+2.5, z_sat[:111]-z_gal[:111]+2.5, 
    #            np.ones(len(x_sat[:111])), color=(1,0,0), line_width=200,
    #            tube_radius=2, opacity=1)



    tt = mlab.contour3d(rho, opacity=0.5, 
                        extent=[-300, 300, 300, 300, -300, 300],
                        transparent=True, colormap='Spectral', vmin=-0.5, vmax=0.6)
    #mlab.colorbar()
    
    scene = engine.scenes[0]
    iso_surface = scene.children[0].children[0].children[0]

    # the following line will print you everything that you can modify on that object
    iso_surface.contour.print_traits()

    # now let's modify the number of contours and the min/max
    # you can also do these steps manually in the mayavi pipeline editor
    iso_surface.compute_normals = False  # without this only 1 contour will be displayed
    iso_surface.contour.number_of_contours = 15
    iso_surface.contour.minimum_contour = 0.6
    iso_surface.contour.maximum_contour = -0.5
    
    lut =tt.module_manager.scalar_lut_manager.lut.table.to_array()
    ilut = lut[::-1]
    # putting LUT back in the surface object
    tt.module_manager.scalar_lut_manager.lut.table = ilut
    #tt.actor.actor.rotate_z(270)
    #mlab.roll(270)
    #limits
    


    
    mlab.view(azimuth=angle, distance=1200)
    
    #zz = mlab.plot3d(xlmc, ylmc, zlmc,
    #                 np.ones(len(xlmc))*200, line_width=200, tube_radius=2, color=(1,1,1), opacity=1)
    
    mlab.savefig(figname, size=(400,400))
    mlab.close()
    engine.stop()

    
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
    xlmc = x_sat[:111]-x_gal[:111]#+2.5
    ylmc = y_sat[:111]-y_gal[:111]#+2.5
    zlmc = z_sat[:111]-z_gal[:111]#+2.5

    nbins = 120
    #densities = 'rho_mwlmc_bfe.txt'
    densities1 = 'rhomwlmc.txt'
    densities2 = 'rhomwlmc_base.txt'

    #densities = 'mw_bfe_density_SN_0_snap_000.txt'
    rho1 = load_density(densities1, nbins)
    rho2 = load_density(densities2, nbins)
    
    for i in range(360, 361):
        figname = '../movies/mwlmc_rotation/wake_mwlmc_angle_{:0>3d}.png'.format(i)
        mayavi_plot(xlmc, ylmc, zlmc, rho1/rho2 - 1, 90, figname)



#for i in range(0, 360):
#    mlab.view(azimuth=i, elevation=90, distance=1200)
 




"""
r_s_sims = 40.85

coeff_c = np.loadtxt('../../SCF_tools/PCA/MWLMC5_coeff_20_20_100M_b1.txt')
S = coeff_c[:,0]
T = coeff_c[:,1]

S_matrix = np.zeros((21, 21, 21))
T_matrix = np.zeros((21, 21, 21))


counter = 0
for n in range(21):
    for l in range(21):
        for m in range(0, l+1):
            S_matrix[n][l][m] = S[counter]
            T_matrix[n][l][m] = T[counter]
            counter +=1
            
x_grid = np.arange(-300, 300, 5)
y_grid = np.arange(-300, 300, 5)
z_grid = np.arange(-300, 300, 5)
x_grid, y_grid, z_grid = np.meshgrid(x_grid, y_grid, z_grid)


rho_mwlmc = biff.density(np.ascontiguousarray(np.double(np.array([x_grid.flatten(), 
                                                        y_grid.flatten(),
                                                        z_grid.flatten()]).T)), 
                           S_matrix, 
                           T_matrix,
                           M=1, r_s=r_s_sims)
S_mwlmc_000 = np.zeros(np.shape(S_matrix))
S_mwlmc_000[0,0,0] = S_matrix[0,0,0]
T_mwlmc_000 = np.zeros(np.shape(T_matrix))
T_mwlmc_000[0,0,0] = T_matrix[0,0,0]


np.savetxt('rhomwlmc.txt', rho_mwlmc)

rho_mwlmc_base = biff.density(np.ascontiguousarray(np.double(np.array([x_grid.flatten(), y_grid.flatten(), 
                                                                         z_grid.flatten()]).T)), 
                           S_mwlmc_000, 
                           T_mwlmc_000,
                           M=1, r_s=r_s_sims)


np.savetxt('rhomwlmc_base.txt', rho_mwlmc_base)


"""
