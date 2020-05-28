import numpy as np
import gala.potential.scf as gp

def density_field(S, T, rs, m, xmin=-50, xmax=50, 
        ymin=-50, ymax=50, dx=1.5, dy=1.5, field='density'):

    """


    """
    # meshgrid
    x_grid = np.arange(xmin, xmax, dx)
    y_grid = np.arange(ymin, ymax, dy)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    xyz = np.array([np.zeros(len(y_grid.flatten())), 
                    x_grid.flatten(), 
                    y_grid.flatten()])

    # compute density
    pot = gp.SCFPotential(m=m, r_s=rs, Snlm=S, Tnlm=T)
    if field=='density':
        rho = pot.density(xyz)
    elif field=='potential':
        rho = pot.energy(xyz)

    return rho

