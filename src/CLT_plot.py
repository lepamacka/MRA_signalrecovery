import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

def lklhd(x_abs_k, z, M, k):
    if k == 0 or (k == z.shape[-1]//2 and z.shape[-1]%2 == 0):
        k_fac = 1
    else:
        k_fac = 1

    res = np.exp(-1. * ((z[k] - M * (1. + (x_abs_k**2))) / np.sqrt(2. * (M * (1. + 2. * (x_abs_k**2)))))**2 / 2.)
    res *= k_fac / (np.sqrt(2. * np. pi))

    return res

M_true = 10000
signal_true = np.array([3., 0., 0.])
x_abs = np.abs(fft(signal_true, norm='ortho'))
print(x_abs)
z_true = M_true * (x_abs ** 2)

Xs = np.arange(0.1, 5, 0.01)
Ys = np.arange(0.1, 5, 0.01)
X, Y = np.meshgrid(Xs, Ys, indexing='xy')
Z = np.log(lklhd(X, z_true, M_true, 0) * lklhd(Y, z_true, M_true, 1))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(X, Y, Z, linewidth=0)

ax.set_xlabel("|x_tilde[0]|")
ax.set_ylabel("|x_tilde[1]|")
plt.show()