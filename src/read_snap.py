import numpy as np
from pygadgetreader import readsnap
import sys

def load_snapshot(snapname, snapformat, masscol=3):
    if snapformat == 1:
        pos = readsnap(snapname, 'pos', 'dm')
        mass = readsnap(snapname, 'mass', 'dm')

    elif snapformat == 2:
        snap = np.loadtxt(snapname)
        pos = snap[:,0:3]
        mass = snap[:,masscol]

    else : 
        print('Wrong snapshot format: (1) Gadget, (2) ASCII')
        sys.exit()

    return np.ascontiguousarray(pos), np.ascontiguousarray(mass)
