import numpy as np
import matplotlib.pyplot as plt


def scatter_plot(snap, data):
    rand = np.random.randint(0, len(data), 50000)
    x = data[rand,0]
    y = data[rand,1]
    z = data[rand,2]

    plt.scatter(x,y,s=0.1, c='k')
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.savefig(snap+"_xy.png", bbox_inches='tight')
    plt.close()


    plt.scatter(y,z,s=0.1, c='k')
    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    plt.savefig(snap+"_yz.png", bbox_inches='tight')
    plt.close()


def density_plot(snap, data):
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    r = np.sqrt(np.sum(data**2, axis=1))
    index_cut = np.where((r<300) & (np.abs(x<5)))[0]
    hist_slice = np.histogram2d(y[index_cut], z[index_cut], 100)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im1= ax.imshow(np.log10(np.abs(hist_slice[0].T)), origin='lower',  cmap='coolwarm', extent=[-300, 300, -300, 300], vmin=0, vmax=3)
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    plt.savefig(snap+"_yz_density.png", bbox_inches='tight')
