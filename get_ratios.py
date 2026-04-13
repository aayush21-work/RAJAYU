
import fast_loadtxt as fl
import numpy as np
import matplotlib.pyplot as plt
file=fl.loadtxt('background_nmdc.dat',skip_rows=10)

N=file[:,0]
phi=file[:,1]
phip=file[:,2]
phipp=file[:,3]
eps=file[:,4]
H=file[:,5]
Hp=file[:,6]
Hpp=file[:,7]

phidot = phip*H 
phiddot = H*Hp*phip + H**2*phipp

dN= N[1]-N[0]
phippp = np.zeros_like(phipp)
phippp[1:-1] = (phipp[2:] - phipp[:-2]) / (2 * dN)
phidddot=phippp[1:-1]*H[1:-1]

ratio1= (phiddot/phidot)**2
ratio2=phidddot/phidot[1:-1]
# ratio3=phippp[10:-1]/phip[10:-1]

plt.plot(eps[1:-1],ratio1[1:-1])
plt.plot(eps[1:-1],ratio2)
# plt.scatter(eps[1:-1],ratio3)
plt.xlim([0,1e-1])
plt.ylim([0,1])
plt.show()

#np.savetxt("ratios.dat",np.column_stack([eps[1:-1],ratio1[1:-1],ratio2]))




