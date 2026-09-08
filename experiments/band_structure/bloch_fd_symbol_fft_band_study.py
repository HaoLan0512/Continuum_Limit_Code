"""Run the FFT/finite-difference-symbol Bloch-band study.

Method:
    Plane-wave/FFT potential convolution with finite-difference kinetic symbols.
Purpose:
    Compute four bands and plot the potential, selected modes, and perturbations.
Inputs:
    The 256-point FFT/FD-symbol L-BFGS displacement u_min_fft.npy.
Outputs:
    Nine PDF plots for the potential, bands, modes, and displacement comparisons.
Output location:
    outputs/band_structure/output_bloch_fd_symbol_fft_band_study.
Dependencies:
    clr.band_structure.bloch_fd_symbol_fft and the FFT/FD-symbol relaxation
    producer.
Related files:
    The finite-difference band study uses a different solver, grid, and input.
"""

### DESCRIPTION ###

# plots Bloch band structure for the operator - eps^2/2 d/dx^2 + V(x) for a given periodic potential V and k value

### IMPORTS ###

from pathlib import Path

# package imports
import numpy as np
# import sympy as sym
import matplotlib.pyplot as plt

# my functions
from clr.band_structure.bloch_fd_symbol_fft import bloch

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
repository_root = Path(__file__).resolve().parents[2]
u_min_path = (
    repository_root
    / "outputs"
    / "relaxation"
    / "output_fd_symbol_fft_lbfgs_single_case"
    / "u_min_fft.npy"
)
with u_min_path.open('rb') as f:
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
# Create the producer-specific output directory if it doesn't exist.
output_dir = (
    repository_root
    / "outputs"
    / "band_structure"
    / "output_bloch_fd_symbol_fft_band_study"
)
output_dir.mkdir(parents=True, exist_ok=True)
plt.figure()
plt.plot(x, V_per)
plt.xlabel('$x$')
plt.ylabel('$V$')
plt.title('Potential $V$')
plt.savefig(output_dir / (f'phonon_potential_eps_%s.pdf' % eps))

plt.figure()
plt.plot(k_vals, E_vals[:, 0])
plt.xlabel('$k$')
plt.ylabel('$E$')
plt.title('First Bloch band')
plt.savefig(output_dir / (f'first_phonon_band_eps_%s.pdf' % eps))

plt.figure()
for N in range(n_bands):
    plt.plot(k_vals, E_vals[:, N])
plt.xlabel('$k$')
plt.ylabel('$E$')
plt.title('Bloch bands')
plt.savefig(output_dir / (f'phonon_bands_eps_%s.pdf' % eps))

plt.figure()
plt.plot(x, np.real(E_vecs[:, 0, 0]))
plt.xlabel('$x$')
plt.ylabel('$p$')
plt.title(f'Real part of first band periodic Bloch function p(x,k) at k = %s' %
          k_vals[0])
plt.savefig(
    output_dir
    / (f'phonon_mode_first_band_k_%s_eps_%s.pdf' % (k_vals[0], eps))
)

plt.figure()
plt.plot(x, np.real(E_vecs[:, k_gridpoints // 2, 0]))
plt.xlabel('$x$')
plt.ylabel('$p$')
plt.title(f'Real part of first band periodic Bloch function p(x,k) at k = %s' %
          k_vals[k_gridpoints // 2])
plt.savefig(
    output_dir
    / (
        f'phonon_mode_first_band_k_%s_eps_%s.pdf'
        % (k_vals[k_gridpoints // 2], eps)
    )
)

plt.figure()
plt.plot(x, np.real(E_vecs[:, 0, 1]))
plt.xlabel('$x$')
plt.ylabel('$p$')
plt.title(
    f'Real part of second band periodic Bloch function p(x,k) at k = %s' %
    k_vals[0])
plt.savefig(
    output_dir
    / (f'phonon_mode_second_band_k_%s_eps_%s.pdf' % (k_vals[0], eps))
)

plt.figure()
plt.plot(x, np.real(E_vecs[:, k_gridpoints // 2, 1]))
plt.xlabel('$x$')
plt.ylabel('$p$')
plt.title(
    f'Real part of second band periodic Bloch function p(x,k) at k = %s' %
    k_vals[k_gridpoints // 2])
plt.savefig(
    output_dir
    / (
        f'phonon_mode_second_band_k_%s_eps_%s.pdf'
        % (k_vals[k_gridpoints // 2], eps)
    )
)

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
    output_dir
    / (
        f'minimizer_compared_with_minimizer_plus_phonon_first_band_k_%s_eps_%s.pdf'
        % (k_vals[k_gridpoints // 2], eps)
    )
)

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
    output_dir
    / (
        f'minimizer_compared_with_minimizer_plus_phonon_second_band_k_%s_eps_%s.pdf'
        % (k_vals[k_gridpoints // 2], eps)
    )
)

plt.show()
