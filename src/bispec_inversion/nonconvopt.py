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
# Employs PM, nonconvex optimization on phase manifold, as the inversion algorithm.
# SIGNAL MUST BE REAL.
def bispec_inv_PM_real(bispec, W=None, z0=None, z_init=None):
    
    n = np.size(bispec, axis=0)
    
    assert bispec.shape == (n, n)
    
    if W is not None:
        assert W.shape == (n, n)
        bispec = bispec * W * W
    
    if z0 is None:
        z0 = np.sign(bispec[0, 0])
        if z0 == 0:
            z0 = 1.
    else:
        if z0 == 0:
            z0 = 1.
        else: 
            z0 = np.round(np.real(z0)/np.abs(np.real(z0)))

    
    assert z0 == 1. or z0 == -1.
    
    if z_init is None:
        z_init = (2 * np.random.rand(n) - 1) + 1.j * (2 * np.random.rand(n) - 1)
        z_init[0] = z0
    else:
        assert np.shape(z_init) == (n,)
    
    z_init = symmetrize(z_init)
    z = z_init[:(n+1)//2]
    
    manifold = pymanopt.manifolds.complex_circle.ComplexCircle((n+1)//2)
    
    # circ_mat = np.eye(N=n*2, k=0) + np.eye(N=n*2, k=n) + np.eye(N=n*2, k=-n)
    
    if n//2 == n/2:
        
        z_init[n//2] = np.round(np.real(z_init[n//2])/np.abs(np.real(z_init[n//2])))
        
        @pymanopt.function.autograd(manifold)
        def matrix(point):
            concat = anp.concatenate((point, np.array([z_init[n//2]]), anp.conj(point[:0:-1])))
            conv = anp.outer(anp.ones(n), anp.concatenate((concat, concat, np.zeros(1))))
            circ = anp.reshape(conv.ravel()[:2*n*n], (n, 2*n))
            return bispec * anp.conj(circ[:n, n:])
                    
        @pymanopt.function.autograd(manifold)
        def cost(point):
            concat = anp.concatenate((point, np.array([z_init[n//2]]), anp.conj(point[:0:-1])))
            res = anp.conj(concat) @ (matrix(point) @ concat)
            return -anp.real(res) / (n ** 2)

    else:
        @pymanopt.function.autograd(manifold)
        def matrix(point):
            concat = anp.concatenate((point, anp.conj(point[:0:-1])))
            mat = anp.outer(anp.ones(n), anp.concatenate((concat, concat, np.zeros(1))))
            circ = anp.reshape(mat.ravel()[:2*n*n], (n, 2*n))
            return bispec * anp.conj(circ[:n, n:])
                    
        @pymanopt.function.autograd(manifold)
        def cost(point):
            concat = anp.concatenate((point, anp.conj(point[:0:-1])))
            res = anp.conj(concat) @ (matrix(point) @ concat)
            return -anp.real(res) / (n ** 2)
                
    problem = pymanopt.Problem(manifold, cost)
    
    optimizer = pymanopt.optimizers.trust_regions.TrustRegions(verbosity=0)
    
    z_opt = optimizer.run(problem, initial_point=z).point
    
    if n//2 == n/2:
        cost1 = cost(z_opt)
        mid1 = z_init[n//2]
        mid2 = -z_init[n//2]
        z_init[n//2] = mid2
        z_opt2 = optimizer.run(problem, initial_point=z).point
        if cost(z_opt2) >= cost1:
            z_opt = np.concatenate((z_opt, np.array([mid1]), np.conj(z_opt[:0:-1])))
        else:
            z_opt = np.concatenate((z_opt2, np.array([mid2]), np.conj(z_opt2[:0:-1])))
    else:
        z_opt = np.concatenate((z_opt, np.conj(z_opt[:0:-1])))

    z_opt = z_opt * (z_init[0] / z[0])
    z_opt = symmetrize(z_opt)
        
    return z_opt

# Inverts the bispectrum to obtain phases for dft of corresponding signal.
# Employs PM, nonconvex optimization on phase manifold, as the inversion algorithm.
# SIGNAL MUST BE COMPLEX.
def bispec_inv_PM_complex(bispec, W=None, z0=None, z_init=None):
    
    n = np.size(bispec, axis=0)
    
    assert bispec.shape == (n, n)
    
    if W is not None:
        assert W.shape == (n, n)
        bispec = bispec * W * W
    
    if z0 is None:
        if np.abs(bispec[0, 0]) == 0.:
            z0 = 1.
        else:
            z0 = bispec[0, 0]/np.abs(bispec[0, 0])
    else:
        if z0 == 0:
            z0 = 1.
        else: 
            z0 = z0/np.abs(z0)
    
    if z_init is None:
        z_init = (2 * np.random.rand(n) - 1) + 1.j * (2 * np.random.rand(n) - 1)
        z_init[0] = z0
    else:
        assert np.shape(z_init) == (n,)
    
    z = z_init
    
    manifold = pymanopt.manifolds.complex_circle.ComplexCircle(n)
    
    @pymanopt.function.autograd(manifold)
    def matrix(point):
        mat = anp.outer(anp.ones(n), anp.concatenate((point, point, np.zeros(1))))
        circ = anp.reshape(mat.ravel()[:2*n*n], (n, 2*n))
        return bispec * anp.conj(circ[:n, n:])
                
    @pymanopt.function.autograd(manifold)
    def cost(point):
        return -anp.real(anp.conj(point) @ (matrix(point) @ point)) / (n ** 2)
                
    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.trust_regions.TrustRegions(verbosity=0)
    z_opt = optimizer.run(problem, initial_point=z).point

    z_opt = z_opt * (z_init[0] / z[0])
        
    return z_opt

    
if __name__ == '__main__':
    
    N = 20
    N = N * 2 + 1
    signal_true = np.zeros(N, dtype=np.complex128)
    # signal_true[0:(N-1)//2] = 1.
    # signal_true[0:(N-1)//2] = np.sin(np.arange((N-1)//2)*6*np.pi/N)
    signal_true[0:(N-1)//2] = np.linspace(0, 1, num=(N-1)//2)
    # signal_true[0:(N-1)//2] = 1 + np.linspace(0, 4, num=(N-1)//2) * 1j
    cmplx = np.any(np.iscomplex(signal_true))
    
    signal_mean = np.mean(signal_true)
    signal_fft = fft(signal_true)
    signal_pwrspec = signal_fft * np.conj(signal_fft)
    signal_bispec = (np.outer(signal_fft, np.conj(signal_fft))) * (circulant(signal_fft))
    signal_adj_fft = fft(signal_true - signal_mean)
    signal_adj_bispec = np.outer(signal_adj_fft, np.conj(signal_adj_fft)) * circulant(signal_adj_fft)
    
    # test = np.array([1, 2, 3, 4])
    # mat_test = anp.outer(anp.ones(4), anp.concatenate((test, test, np.zeros(1))))
    # circ_test = anp.reshape(mat_test.ravel()[:2*4*4], (4, 2*4))
    # print(circ_test[:4, 4:])
    
    if cmplx:
        phases_est = bispec_inv_PM_complex(signal_adj_bispec, z0=signal_mean)
    else:
        phases_est = bispec_inv_PM_real(signal_adj_bispec, z0=signal_mean)
    
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
    
    
    
    
    

    
    
    