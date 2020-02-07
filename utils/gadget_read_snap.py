"""
Code to read Gagdet binary outputs.


"""

import numpy as np


def read_snap(filename):
    f = open(filename,'rb')    
    blocksize = np.fromfile(f,dtype=np.int32,count=1)[0]
    print(blocksize)
    bytesleft=96
    npart=np.fromfile(f, dtype=np.int32, count=6)
    massarr = np.fromfile(f,dtype=np.float64,count=6)
    time = (np.fromfile(f,dtype=np.float64,count=1))[0]
    redshift = (np.fromfile(f,dtype=np.float64,count=1))[0]
    sfr=(np.fromfile(f,dtype=np.int32,count=1))[0]
    feedback = (np.fromfile(f,dtype=np.int32,count=1))[0]
    nall = np.fromfile(f,dtype=np.int32,count=6)
    cooling = (np.fromfile(f,dtype=np.int32,count=1))[0]
    filenum = (np.fromfile(f,dtype=np.int32,count=1))[0]
    boxsize = (np.fromfile(f,dtype=np.float64,count=1))[0]
    omega_m = (np.fromfile(f,dtype=np.float64,count=1))[0]
    omega_l = (np.fromfile(f,dtype=np.float64,count=1))[0]
    hubble = (np.fromfile(f,dtype=np.float64,count=1))[0]
    headerend = np.fromfile(f, dtype=np.int8, count=96)
    blocksize = np.fromfile(f,dtype=np.int32,count=1)[0]
    #print headerend
    #print blocksize

    blocksize = np.fromfile(f,dtype=np.int32,count=1)[0]
    #print blocksize, 'blocksize pp'
    #ntot=np.int64(np.sum(npart))
    dt = np.dtype((np.float32,3)) # type of entry for POS, VEL, ACCEL, etc.
    pp=np.fromfile(f, dtype=np.dtype(dt), count=np.sum(npart))
    #print pp, np.size(pp)
    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]
    print(blocksize, 'blocksize pp')


    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]
    #print blocksize, 'blocksize vv'
    vv=np.fromfile(f, dtype=dt, count=np.sum(npart))
    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]
    #print blocksize, 'blocksize vv'


    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]
    #print blocksize, 'blocksize ids'
    ids=np.fromfile(f, dtype=np.uint32, count=np.sum(npart))
    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]
    print(blocksize, 'blocksize ids')

    #print ids, 'ids'
    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]
    mass=np.fromfile(f, dtype=np.float32, count=np.sum(npart))
    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]

    print(blocksize, 'blocksize mass')


    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]
    #print blocksize, 'blocksize vv'
    pot=np.fromfile(f, dtype=np.float32, count=np.sum(npart))
    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]


    blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]
    #print blocksize, 'blocksize vv'
    acc=np.fromfile(f, dtype=dt, count=np.sum(npart))
    #blocksize = np.fromfile(f,dtype=np.uint32,count=1)[0]
    
    return pp, pot, acc, ids
