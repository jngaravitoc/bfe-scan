import numpy as np
import sys
sys.path.append('../bfe-py/')
from bfe import com

def test_com_halo():
    filename = "/home/xzk/work/github/bfe-py/tests/data/plummer_sphere_10K.txt"
    data = np.loadtxt(filename)
    pos = data[:,0:3]
    vel = data[:,3:6]
    mk = data[:,6]
    rcom, vcom = com.COM(pos, vel, mk)
    rcom_true = np.array([ 7.46023226, 29.95142926, -5.48522253])
    vcom_true = np.array([ 11.74871876,  41.78031904, -65.730021])

    assert np.allclose(rcom, rcom_true, rtol=1e-2)
    assert np.allclose(vcom, vcom_true, rtol=1e-2)


def test_com_shrinking_sphere_halo():
    filename = "/home/xzk/work/github/bfe-py/tests/data/plummer_sphere_10K.txt"
    data = np.loadtxt(filename)
    pos = data[:,0:3]
    vel = data[:,3:6]
    mk = data[:,6]
    rcom, vcom = com.CM(pos, vel, mk, delta=0.025)
    rcom_true = np.array([-0.05291256, -0.10510422,  0.18606588])
    vcom_true = np.array([-3.63038043,  0.33830252,  2.06705637])

    assert np.allclose(rcom, rcom_true, rtol=1e-2), \
            """bfe-py.com.shrinking_sphere is failing """
    assert np.allclose(vcom, vcom_true, rtol=1e-2), \
            """bfe-py.com.shrinking_sphere is failing """


