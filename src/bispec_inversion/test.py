# import numpy as np
# from scipy.fft import fft
# from scipy.linalg import circulant
#import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt

# N = 10
# N = N*2 + 1
# M = 10**4
# batch = 100

# signal = np.zeros(N)
# signal[0:N//2] = 1.

# shifts = np.random.randint(N, size=batch)

# signal_circ = np.transpose(circulant(signal))
# print(signal_circ)
# print(signal_circ[shifts, :])

anp.random.seed(42)

dim = 3
manifold = pymanopt.manifolds.Sphere(dim)

matrix = anp.random.normal(size=(dim, dim))
matrix = 0.5 * (matrix + matrix.T)

@pymanopt.function.autograd(manifold)
def cost(point):
    return -point @ matrix @ point

problem = pymanopt.Problem(manifold, cost)

optimizer = pymanopt.optimizers.SteepestDescent()
result = optimizer.run(problem)

eigenvalues, eigenvectors = anp.linalg.eig(matrix)
dominant_eigenvector = eigenvectors[:, eigenvalues.argmax()]

print("Dominant eigenvector:", dominant_eigenvector)
print("Pymanopt solution:", result.point)