import numpy as np
import matplotlib.pyplot as plt

length = 101

signal_1 = np.zeros((length,))
signal_1[length//4:(length//4)+(length//8)] += np.linspace(0, 1, length//8)
signal_1[(length//4)+(length//8):(length//4)+2*(length//8)] += np.linspace(1, 0, length//8)

signal_2 = np.zeros((length,))
signal_2[2*length//3:(2*length//3)+(length//8)] += np.linspace(0, 1, length//8)
signal_2[2*length//3+(length//8):(2*length//3)+2*(length//8)] += np.linspace(1, 0, length//8)


sigma_low = 0.1
sample_1_low = (signal_1 + sigma_low * np.random.randn(length)) / (1 + sigma_low**2)
sample_2_low = (signal_2 + sigma_low * np.random.randn(length)) / (1 + sigma_low**2)

sigma_high = 1.
sample_1_high = (signal_1 + sigma_high * np.random.randn(length)) / (1 + sigma_high**2)
sample_2_high = (signal_2 + sigma_high * np.random.randn(length)) / (1 + sigma_high**2)

# gridspec_kw = {"hspace": -0.1, "wspace": -0.2}
fig, axes = plt.subplots(3, 2, squeeze=True, sharey='all', sharex='all')

axes[0, 0].plot(signal_1, c="orange", linewidth=1.5)
axes[0, 1].plot(signal_2, c="orange", linewidth=1.5)
axes[1, 0].plot(sample_1_low, c="orange", linewidth=1.5)
axes[1, 1].plot(sample_2_low, c="orange", linewidth=1.5)
axes[2, 0].plot(sample_1_high, c="orange", linewidth=1.5)
axes[2, 1].plot(sample_2_high, c="orange", linewidth=1.5)

for axis in axes.flatten():
    axis.set_aspect(10.)
    axis.set_axis_off()

fig.tight_layout(pad=1., h_pad=-14., w_pad=0., rect=None)

plt.savefig("./../figs/minor_plots/MRA_noise.png")
plt.savefig("./../figs/minor_plots/MRA_noise.pdf")

plt.show()