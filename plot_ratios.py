import fast_loadtxt as fl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.labelsize": 14,
        "lines.linewidth": 1.5,
        "legend.fontsize": 12,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


file = fl.loadtxt('background_nmdc.dat', skip_rows=10)
N    = file[:,0]
phip = file[:,2]
phipp= file[:,3]
eps  = file[:,4]
H    = file[:,5]
Hp   = file[:,6]

phidot  = phip * H
phiddot = H * Hp * phip + H**2 * phipp

dN = N[1] - N[0]
phippp = np.zeros_like(phipp)
phippp[1:-1] = (phipp[2:] - phipp[:-2]) / (2 * dN)
phidddot = phippp[1:-1] * H[1:-1]

ratio1 = (phiddot / phidot)**2
ratio2 = phidddot / phidot[1:-1]

fig, ax = plt.subplots(figsize=(5, 3.5))

ax.plot(eps[1:-1], ratio1[1:-1],
        color='#8B3A3A', lw=1.2,
        label=r'$(\ddot{\phi}/\dot{\phi})^2$')
ax.plot(eps[1:-1], ratio2,
        color='#3A5F8B', lw=1.2,
        label=r'$\dddot{\phi}/\dot{\phi}$')

ax.set_xlim([0, 1e-1])
ax.set_ylim([0, 1])
ax.set_xlabel(r'$\varepsilon$')
ax.set_ylabel('Ratio')
ax.legend(frameon=False)

fig.tight_layout(pad=0.4)
fig.savefig('slowroll_ratios.pdf', bbox_inches='tight')
fig.savefig('slowroll_ratios.png', dpi=600, bbox_inches='tight')
plt.show()
