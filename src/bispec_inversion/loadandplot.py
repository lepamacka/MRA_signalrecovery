import numpy as np
import matplotlib.pyplot as plt
from helperfuncs import fft_stats
from generateandsave import load_stats_from_file

def signalplotter(IPS_data_path, PM_data_path, plotname, plottitle):
    
    signal_true, dft_PM, pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_PM = load_stats_from_file(PM_data_path)
    _, dft_IPS, pwrspec_IPS, bispec_IPS, bispec_adj_IPS, relerr_IPS, signal_IPS, time_IPS = load_stats_from_file(IPS_data_path)
    
    N = len(signal_true)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(range(N), signal_true, 'k*-')
    ax.plot(range(N), signal_PM, 'go-.')
    ax.plot(range(N), signal_IPS, 'rs--')
    
    ax.legend(('True', 'PM', 'IPS'))
    ax.set_title(plottitle)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Component')
    
    fig.savefig(plotname, format='pdf')
    

def relerrplotter_dataptseq(IPS_data_path, PM_data_path, sequence, plotname, plottitle):
    
    signal_true, dft_PM, pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_PM = load_stats_from_file(PM_data_path)
    _, dft_IPS, pwrspec_IPS, bispec_IPS, bispec_adj_IPS, relerr_IPS, signal_IPS, time_IPS = load_stats_from_file(IPS_data_path)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.loglog(sequence, relerr_PM, 'go-.')
    ax.loglog(sequence, relerr_IPS, 'rs--')
    
    ax.legend(('PM', 'IPS'))
    ax.set_title(plottitle)
    ax.set_ylabel('Relative error for signal')
    ax.set_xlabel('Datapoints')
    
    fig.savefig(plotname, format='pdf')

def relerrplotter_sigmaseq(IPS_data_path, PM_data_path, sequence, plotname, plottitle):
    
    signal_true, dft_PM, pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_PM = load_stats_from_file(PM_data_path)
    _, dft_IPS, pwrspec_IPS, bispec_IPS, bispec_adj_IPS, relerr_IPS, signal_IPS, time_IPS = load_stats_from_file(IPS_data_path)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.loglog(sequence, relerr_PM, 'go-.')
    ax.loglog(sequence, relerr_IPS, 'rs--')
    
    ax.legend(('PM', 'IPS'))
    ax.set_title(plottitle)
    ax.set_ylabel('Relative error for signal')
    ax.set_xlabel('Sigma')
    
    fig.savefig(plotname, format='pdf')
    
def bipwrspecplotter(PM_data_path, sequence, plotname, plottitle):
    
    signal_true, dft_PM, pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_PM = load_stats_from_file(PM_data_path)

    true_mean, true_fft, true_pwrspec, true_bispec = fft_stats(signal_true)
    
    pwrspec_err = true_pwrspec - pwrspec_PM
    bispec_err = true_bispec - bispec_PM
    
    pwrspec_relerr = np.linalg.norm(pwrspec_err, axis=1)/np.linalg.norm(true_pwrspec)
    bispec_relerr = np.linalg.norm(bispec_err, axis=(1, 2))/np.linalg.norm(true_bispec)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.loglog(sequence, pwrspec_relerr, 'bo-.')
    ax.loglog(sequence, bispec_relerr, 'ms--')
    
    ax.legend(('Power Spectrum', 'Bispectrum'))
    ax.set_title(plottitle)
    ax.set_ylabel('Relative error')
    ax.set_xlabel('Datapoints')
    
    fig.savefig(plotname, format='pdf')
    

if __name__ == '__main__':
    
    # signal_true, dft_PM, pwrspec_PM, bispec_PM, bispec_adj_PM, relerr_PM, signal_PM, time_PM = load_stats_from_file('PM_data/PM_gauss_ptsseq_100_10mil_sigma_1.npy')
    # _, dft_IPS, pwrspec_IPS, bispec_IPS, bispec_adj_IPS, relerr_IPS, signal_IPS, time_IPS = load_stats_from_file('IPS_data/IPS_gauss_ptsseq_100_10mil_sigma_1.npy')
    
    batch_seq = (10 ** 2) * np.round(10 ** np.linspace(0, 5, 11)).astype(np.int32)
    sigma_seq = 10 ** np.linspace(-1, 1, 9)
    
    with open('IPS_data_paths.txt', 'rb') as g:
        f = g.readlines()
        IPS_gauss_pts = f[0][:-2]
        IPS_gauss_sigmas = f[1][:-2]
        IPS_semisin_pts = f[2][:-2]
        IPS_semisin_sigmas = f[3][:-2]
        IPS_step_pts = f[4][:-2]
        IPS_step_sigmas = f[5]
        
    with open('PM_data_paths.txt', 'rb') as g:
        f = g.readlines()
        PM_gauss_pts = f[0][:-2]
        PM_gauss_sigmas = f[1][:-2]
        PM_semisin_pts = f[2][:-2]
        PM_semisin_sigmas = f[3][:-2]
        PM_step_pts = f[4][:-2]
        PM_step_sigmas = f[5]

    signalplotter(IPS_gauss_pts, PM_gauss_pts, 'gauss_signal.pdf', 'Signal = Gauss, Sigma = 1')
    signalplotter(IPS_step_pts, PM_step_pts, 'step_signal.pdf', 'Signal = Step, Sigma = 1')
    signalplotter(IPS_semisin_pts, PM_semisin_pts, 'semisin_signal.pdf', 'Signal = Semi-Sine, Sigma = 1')
    
    relerrplotter_dataptseq(IPS_gauss_pts, PM_gauss_pts, batch_seq, 'gauss_signal_relerr_dataptseq.pdf', 'Signal = Gauss, Sigma = 1')
    relerrplotter_dataptseq(IPS_step_pts, PM_step_pts, batch_seq, 'step_signal_relerr_dataptseq.pdf', 'Signal = Step, Sigma = 1')
    relerrplotter_dataptseq(IPS_semisin_pts, PM_semisin_pts, batch_seq, 'semisin_signal_relerr_dataptseq.pdf', 'Signal = Semi-Sine, Sigma = 1')
    
    relerrplotter_sigmaseq(IPS_gauss_sigmas, PM_gauss_sigmas, sigma_seq, 'gauss_signal_relerr_sigmaseq.pdf', 'Signal = Gauss, Datapoints = 10^5')
    relerrplotter_sigmaseq(IPS_step_sigmas, PM_step_sigmas, sigma_seq, 'step_signal_relerr_sigmaseq.pdf', 'Signal = Step, Datapoints = 10^5')
    relerrplotter_sigmaseq(IPS_semisin_sigmas, PM_semisin_sigmas, sigma_seq, 'semisin_signal_relerr_sigmaseq.pdf', 'Signal = Semi-Sine, Datapoints = 10^5')
    
    bipwrspecplotter(PM_gauss_pts, batch_seq, 'gauss_bipwrspec_relerr_dataptseq_sigma1.pdf', 'Signal = Gauss, Sigma = 1')
    bipwrspecplotter(PM_step_pts, batch_seq, 'step_bipwrspec_relerr_dataptseq_sigma1.pdf', 'Signal = Step, Sigma = 1')
    bipwrspecplotter(PM_semisin_pts, batch_seq, 'semisin_bipwrspec_relerr_dataptseq_sigma1.pdf', 'Signal = Semi-Sine, Sigma = 1')
    
    bigsigma_step_pts = 'PM_step_ptsseq_100_10mil_sigma_10.npy'
    bipwrspecplotter(bigsigma_step_pts, batch_seq, 'step_bipwrspec_relerr_dataptseq_sigma10.pdf', 'Signal = Step, Sigma = 10')
    
    # fig, ax = plt.subplots()
    # ax.loglog(batch_size*batch_seq, relerr_PM, 'go--')
    # ax.loglog(batch_size*batch_seq, relerr_IPS, 'rs--')