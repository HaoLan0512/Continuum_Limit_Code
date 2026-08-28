### DESCRIPTION ###

# plots Bloch band structure for the operator - eps^2/2 d/dx^2 + V(x) for a given periodic potential V and k value

### IMPORTS ###

# package imports
import numpy as np
# import sympy as sym
import matplotlib.pyplot as plt

# my functions
from bloch_fft import bloch
import os

### PARAMETERS ###

# lattice constant
a = 2 * np.pi

# endpoints of x and k grids
x0, x1 = 0, a
k0, k1 = -np.pi / a, np.pi / a

# number of x gridpoints
x_gridpoints = 256

# number of k points
k_gridpoints = 128

# bands to plot
bands = np.array([0, 1, 2, 3])

# epsilon
eps = .1

### MAIN ###

# generate x grid
x = np.linspace(x0, x1, x_gridpoints, endpoint=False)

# load relaxed profile
u_min_path = os.path.join(os.path.dirname(__file__), 'u_min_fft.npy')
with open(u_min_path, 'rb') as f:
    u_min = np.load(f)
# generate phonon potential
V_per = (1 / 2) * np.cos(x + u_min)

# loop over k and band index
k_vals = np.linspace(k0, k1, k_gridpoints, endpoint=False)
n_bands = bands.size
E_vals = np.zeros((k_gridpoints, n_bands))
E_vecs = np.zeros((x_gridpoints, k_gridpoints, n_bands), dtype="complex")
for N in range(k_gridpoints):
    # compute Bloch function evaluated on grid
    vals, vecs = bloch(V_per, x, eps, k_vals[N])
    for band_index in bands:
        E_vals[N, band_index] = vals[band_index]
        E_vecs[:, N, band_index] = vecs[:, band_index]

### OUTPUT ###
# Create output directory if it doesn't exist
output_dir = "result_fft"
os.makedirs(output_dir, exist_ok=True)

# Change working directory to output directory for saving plots
os.chdir(output_dir)
plt.figure()
plt.plot(x, V_per)
plt.xlabel('$x$')
plt.ylabel('$V$')
plt.title('Potential $V$')
plt.savefig(f'phonon_potential_eps_%s.pdf' % eps)

plt.figure()
plt.plot(k_vals, E_vals[:, 0])
plt.xlabel('$k$')
plt.ylabel('$E$')
plt.title('First Bloch band')
plt.savefig(f'first_phonon_band_eps_%s.pdf' % eps)

plt.figure()
for N in range(n_bands):
    plt.plot(k_vals, E_vals[:, N])
plt.xlabel('$k$')
plt.ylabel('$E$')
plt.title('Bloch bands')
plt.savefig(f'phonon_bands_eps_%s.pdf' % eps)

plt.figure()
plt.plot(x, np.real(E_vecs[:, 0, 0]))
plt.xlabel('$x$')
plt.ylabel('$p$')
plt.title(f'Real part of first band periodic Bloch function p(x,k) at k = %s' %
          k_vals[0])
plt.savefig(f'phonon_mode_first_band_k_%s_eps_%s.pdf' % (k_vals[0], eps))

plt.figure()
plt.plot(x, np.real(E_vecs[:, k_gridpoints // 2, 0]))
plt.xlabel('$x$')
plt.ylabel('$p$')
plt.title(f'Real part of first band periodic Bloch function p(x,k) at k = %s' %
          k_vals[k_gridpoints // 2])
plt.savefig(f'phonon_mode_first_band_k_%s_eps_%s.pdf' %
            (k_vals[k_gridpoints // 2], eps))

plt.figure()
plt.plot(x, np.real(E_vecs[:, 0, 1]))
plt.xlabel('$x$')
plt.ylabel('$p$')
plt.title(
    f'Real part of second band periodic Bloch function p(x,k) at k = %s' %
    k_vals[0])
plt.savefig(f'phonon_mode_second_band_k_%s_eps_%s.pdf' % (k_vals[0], eps))

plt.figure()
plt.plot(x, np.real(E_vecs[:, k_gridpoints // 2, 1]))
plt.xlabel('$x$')
plt.ylabel('$p$')
plt.title(
    f'Real part of second band periodic Bloch function p(x,k) at k = %s' %
    k_vals[k_gridpoints // 2])
plt.savefig(f'phonon_mode_second_band_k_%s_eps_%s.pdf' %
            (k_vals[k_gridpoints // 2], eps))

plt.figure()
plt.plot(x, x + u_min, label='minimizer')
plt.plot(x,
         x + u_min + E_vecs[:, k_gridpoints // 2, 0],
         label='minimizer + phonon mode')
plt.plot(x,
         x + u_min - E_vecs[:, k_gridpoints // 2, 0],
         label='minimizer + phonon mode half period')
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend(loc='upper left')
plt.title(
    f'Real part of $u$ and $u + p$ first band periodic Bloch function p(x,k) at k = %s'
    % k_vals[k_gridpoints // 2])
plt.savefig(
    f'minimizer_compared_with_minimizer_plus_phonon_first_band_k_%s_eps_%s.pdf'
    % (k_vals[k_gridpoints // 2], eps))

plt.figure()
plt.plot(x, x + u_min, label='minimizer')
plt.plot(x,
         x + u_min + E_vecs[:, k_gridpoints // 2, 1],
         label='minimizer + phonon mode')
plt.plot(x,
         x + u_min - E_vecs[:, k_gridpoints // 2, 1],
         label='minimizer + phonon mode half period')
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend(loc='upper left')
plt.title(
    f'Real part of $u$ and $u + p$ second band periodic Bloch function p(x,k) at k = %s'
    % k_vals[k_gridpoints // 2])
plt.savefig(
    f'minimizer_compared_with_minimizer_plus_phonon_second_band_k_%s_eps_%s.pdf'
    % (k_vals[k_gridpoints // 2], eps))

plt.show()
