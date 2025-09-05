import numpy as np
from scipy.fft import fft, ifft
import autograd.numpy as anp
import pymanopt
import matplotlib.pyplot as plt

def circulant(x):
    
    dim = np.size(x, axis=0)
    
    assert np.shape(x) == (dim,)
    
    circ = np.zeros(shape=(dim, dim), dtype=x.dtype)
    
    for k in range(dim):
        circ[k, :] += np.roll(x, k, axis=0)
        
    return circ

def symmetrize(y):
    y[1:] += np.conj(np.flip(y[1:]))
    y[1:] *= 0.5
    return y / np.abs(y)

# Inverts the bispectrum to obtain phases for dft of corresponding signal.
# Employs IPS, iterated phase synchronization, as the inversion algorithm.
# SIGNAL MUST BE REAL.
def bispec_inv_IPS_real(bispec, z0=None, z_init=None, maxiter=100):
    
    bispec = bispec/np.abs(bispec)
    
    n = np.size(bispec, axis=0)
    
    if z0 is None:
        z0 = np.sign(bispec[0, 0])
        if z0 == 0:
            z0 = 1.
    else:
        if z0 == 0:
            z0 = 1.
        else: 
            z0 = np.round(np.real(z0)/np.abs(np.real(z0)))
    
    if z_init is None:
        z_init = (2 * np.random.rand(n) - 1) + 1.j * (2 * np.random.rand(n) - 1)
        z_init[0] = z0
    else:
        assert np.shape(z_init) == (n,)
        if z_init[0] != 1. and z_init[0] != -1.:
            z_init[0] = z0
    
    z_init = symmetrize(z_init)
    
    assert z_init[0] == 1. or z_init[0] == -1.
    
    manifold = pymanopt.manifolds.complex_circle.ComplexCircle(n)
    z = z_init
    
    for i in range(maxiter):
        
        matrix = bispec * np.conj(circulant(z))
            
        @pymanopt.function.autograd(manifold)
        def cost(point):
            return -anp.real(anp.conj(point) @ (matrix @ point)) / (n ** 2)
            
        problem = pymanopt.Problem(manifold, cost)
        optimizer = pymanopt.optimizers.trust_regions.TrustRegions(verbosity=0)
        z = optimizer.run(problem, initial_point=z).point
            
        z = z * (z_init[0] / z[0])
        z = symmetrize(z)
       
    return z

    
if __name__ == '__main__':
    
    iters = 100
    N = 20
    N = N * 2 +1
    signal_true = np.zeros(N, dtype=np.complex128)
    # signal_true[0:(N-1)//2] = 1.
    signal_true[0:(N-1)//2] = np.linspace(0, 1, num=(N-1)//2)
    # signal_true[0:(N-1)//2] = np.sin(np.arange((N-1)//2)*6*np.pi/N)
    # signal_true[0:(N-1)//2] = 1 + np.linspace(0, 4, num=(N-1)//2) * 1j
    
    # print(signal_true)
    
    signal_mean = np.mean(signal_true)
    signal_fft = fft(signal_true)
    signal_pwrspec = signal_fft * np.conj(signal_fft)
    signal_bispec = (np.outer(signal_fft, np.conj(signal_fft))) * (circulant(signal_fft))
    signal_adj_fft = fft(signal_true - signal_mean)
    signal_adj_bispec = np.outer(signal_adj_fft, np.conj(signal_adj_fft)) * circulant(signal_adj_fft)
    
    #init = fft(signal_true + np.random.normal(0, 0.3, N))
    
    phases_est = bispec_inv_IPS_real(signal_adj_bispec, maxiter=iters)
    
    fft_est = np.sqrt(signal_pwrspec) * phases_est
    signal_est = ifft(fft_est)
    
    min_diff = 100
    min_ind = 0
    for i in range(N):
        if min_diff > np.linalg.norm(signal_true - np.roll(signal_est, i))/np.linalg.norm(signal_true):
            min_diff = np.linalg.norm(signal_true - np.roll(signal_est, i))/np.linalg.norm(signal_true)
            min_ind = i
    
    signal_est = np.roll(signal_est, min_ind)
    
    print(min_diff)
    
    x = np.arange(0, N)
    y1 = signal_true
    y2 = signal_est
    fig, ax = plt.subplots()
    ax.plot(x, y1)
    ax.plot(x, y2)
    
    

    
    
    