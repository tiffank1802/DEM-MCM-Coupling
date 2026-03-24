import numpy as np
import matplotlib.pyplot as plt
from dem2 import sim


# pos_actuel=np.loadtxt('/teamspace/studios/this_studio/DEM/positions.txt')
# plt.plot(pos_init[:,0],"*",label="x")
# plt.plot(pos_init[:,1],"*",label="y")
plt.plot(sim.pos_init[:,2],"*",label="init")
plt.plot(sim.pos[:,2],"*",label="actuel")
plt.title("positions initiales des particules")
plt.legend()
plt.savefig("position_init_vs_acuel.png")