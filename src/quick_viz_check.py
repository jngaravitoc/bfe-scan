import numpy as np
import matplotlib.pyplot as plt
import sys

snap = sys.argv[1]
data = np.loadtxt(snap)

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
