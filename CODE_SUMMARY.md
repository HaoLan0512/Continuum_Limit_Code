# Continuum Limit Code Summary

This directory contains 17 Python scripts and 6 Jupyter notebooks. Its overall numerical workflow is:

`pair potential -> continuum stacking potential W -> relaxed displacement u -> Bloch/phonon bands`

## Relaxation solvers

- `relaxation.py` is the original finite-difference/L-BFGS minimizer for the continuum energy.
- `relaxation _Jv.py` reformulates the paper's functional in terms of `v = x + u`.
- `relaxation_Jv_phase_fixed.py` adds odd symmetry and phase fixing, checks the resulting constraints, and saves the optimized `u` and `v` profiles.
- `1D_fd_fft.py` provides fixed-point and virtual-time solvers using finite-difference Laplacian symbols evaluated in Fourier space.
- `1D_fd_fft_custom.py` adds configurable admissible-set projections, pinning conditions, initial guesses, and comparison with a baseline run.
- `1D_fd_fft_custom_v1.py` additionally records PDE residuals and energy during convergence and implements stronger pinning through `U(x) = sin(x)v(x)`.
- `FFT base/relaxation_fft.py` is an optimization-based FFT/finite-difference-symbol version of the relaxation calculation.

## Bloch-band calculations

- `bloch.py` solves the one-dimensional Bloch eigenvalue problem with finite-difference derivative matrices.
- `main_bands.py` loads a relaxed displacement profile, constructs the periodic potential `V = (1/2) cos(x + u)`, and plots phonon bands and modes.
- `FFT base/bloch.py` is largely a copy of the root finite-difference Bloch solver.
- `FFT base/bloch_fft.py` solves the same problem spectrally in a plane-wave basis.
- `FFT base/main_bands_fft.py` drives the spectral band calculation and produces its plots.

## Lennard-Jones stacking potentials

- `plot_lj_stacking_potential.py` is a compact exploratory implementation of a lattice-summed Lennard-Jones stacking potential and its derivative.
- `build_W_from_pair_potential.py` is the more general command-line generator. It supports monoatomic or diatomic bases, plots `W` and `W'`, and sweeps `(sigma, L)` to classify AA-like versus AB-like minima.
- `build_W_from_LJ.py` is a closely related Lennard-Jones-specific generator with detailed registry and assumption diagnostics.
- `compare_lj_potential_implementations.py` checks agreement between the two potential conventions, including derivatives and AA/AB preference.
- `test.py` is a small sanity check for periodic circular distance.

## Jupyter notebooks

- `1D_FFT.ipynb` explores an FFT/Picard solver for the Euler-Lagrange equation.
- `1D_fd_fft.ipynb` prototypes the finite-difference/FFT solvers later placed in a Python script.
- `1D_JFNK.ipynb` experiments with Newton-Krylov and GMRES for the nonlinear system.
- `1D_split_radix_FFT.ipynb` implements a pure-Python split-radix FFT solver with parameter continuation.
- `one-dimensional moire materials.ipynb` minimizes a discrete one-dimensional GSFE model with L-BFGS-B.
- `FFT base/test.ipynb` tests and compares the Bloch eigensolver implementations.

## Repository character

This is a research and prototyping repository containing several generations of related numerical approaches rather than a single unified application. Some files are direct experiments or near-duplicates, while the newer scripts add constraints, diagnostics, parameter sweeps, and comparison tools.
