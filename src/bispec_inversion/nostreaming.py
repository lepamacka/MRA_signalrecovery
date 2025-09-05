import numpy as np
from scipy.fft import fft
from scipy.linalg import circulant
#import matplotlib.pyplot as plt

np.random.seed(0)

N = 10
N = N*2 + 1
M = 10**5
sigma = 1

signal = np.zeros(N)
signal[0:N//2] = 1.

signal_mean = np.mean(signal)
signal_fft = fft(signal)
signal_pwrspec = signal_fft * np.conj(signal_fft)
signal_bispec = np.outer(signal_fft, np.conj(signal_fft)) * np.transpose(circulant(signal_fft))
#for i in range(N):
#    signal_bispec[i, :] *= np.roll(signal_fft, i)

shifts = np.concatenate(([0], np.sort(np.random.randint(M, size=N-1)), [M]))
data = np.random.normal(0, sigma, size=(M, N))  # + signal[np.newaxis, :]
for i in range(N):
    data[shifts[i]:shifts[i+1], :] += np.roll(signal, i)

sigma_est = np.sqrt(np.var(np.sum(data, 1))/N)

data_mean = np.mean(data)
data_fft = fft(data, axis=1)

data_pwrspec = data_fft * np.conj(data_fft)
data_pwrspec = np.mean(data_pwrspec, 0) - N*(sigma_est**2)

A = np.diag(np.ones(N))
A[0, :] += np.ones(N)
if data.dtype == np.float64:
    A[:, 0] += np.ones(N)
data_bispec = np.matmul(data_fft.reshape(M, N, 1), np.conj(data_fft.reshape(M, 1, N)))
for i in range(N):
    data_bispec[:, i, :] *= np.roll(data_fft, i, axis=1)
data_bispec = np.mean(data_bispec, 0) - (sigma_est**2) * (N**2) * data_mean * A


if __name__ == '__main__':
    print('Sigma difference = ' + str(sigma - sigma_est))
    print('Mean difference = ' + str(signal_mean - data_mean))
    print('Power spectrum mean difference = ' + str(np.mean(signal_pwrspec - data_pwrspec)))
    print('Bispectrum mean difference = ' + str(np.mean(signal_bispec - data_bispec)))
