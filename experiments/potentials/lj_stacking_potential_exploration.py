"""Explore a one-dimensional Lennard-Jones stacking potential.

Method:
    Monoatomic Lennard-Jones 12-6 lattice summation in registry space.
Purpose:
    Plot stacking energies and derivatives and sweep interlayer distances.
Inputs:
    Lattice spacing, interlayer distance, LJ parameters, and lattice-sum truncation.
Outputs:
    Interactive figures only; no files are saved.
Output location:
    None.
Dependencies:
    NumPy and Matplotlib; no local module imports.
Related files:
    Its convention is reproduced independently by
    ``lj_potential_convention_comparison.py``.
"""

import numpy as np
import matplotlib.pyplot as plt


def phi_lj(r, eps_lj=1.0, sigma=1.0):
    """Lennard--Jones 12-6 pair potential."""
    sr6 = (sigma / r) ** 6
    return 4.0 * eps_lj * (sr6**2 - sr6)



def dphi_lj_dr(r, eps_lj=1.0, sigma=1.0):
    """Derivative of phi_lj with respect to r."""
    sr6 = (sigma / r) ** 6
    return 24.0 * eps_lj * (sigma**6) / (r**7) * (1.0 - 2.0 * sr6)



def V_gsfe_delta(delta, a=1.0, L=0.9, eps_lj=1.0, sigma=1.0, nsum=80):
    """
    1D effective stacking energy as a function of horizontal registry delta.

        V_GSFE(delta) = sum_n phi( sqrt((delta - n a)^2 + L^2) ).

    Here delta can be a scalar or a numpy array. The infinite sum is truncated
    to n = -nsum, ..., nsum.
    """
    delta = np.asarray(delta)
    n = np.arange(-nsum, nsum + 1)
    dx = delta[..., None] - n * a
    r = np.sqrt(dx**2 + L**2)
    return np.sum(phi_lj(r, eps_lj=eps_lj, sigma=sigma), axis=-1)



def dV_gsfe_ddelta(delta, a=1.0, L=0.9, eps_lj=1.0, sigma=1.0, nsum=80):
    """Derivative of V_GSFE with respect to delta."""
    delta = np.asarray(delta)
    n = np.arange(-nsum, nsum + 1)
    dx = delta[..., None] - n * a
    r = np.sqrt(dx**2 + L**2)
    return np.sum(dphi_lj_dr(r, eps_lj=eps_lj, sigma=sigma) * dx / r, axis=-1)



def wrap_registry(delta, a=1.0):
    return np.mod(delta, a)



def phase_to_registry(s, a=1.0, cell_shift=0.0):
    """
    Map the 2π-periodic phase variable s to the registry variable delta in [0,a).

    cell_shift lets you relabel which stacking sits at s = 0.
    """
    return wrap_registry(a * s / (2.0 * np.pi) + cell_shift, a=a)



def W_phase_raw(s, a=1.0, L=0.9, eps_lj=1.0, sigma=1.0, nsum=80, cell_shift=0.0):
    delta = phase_to_registry(s, a=a, cell_shift=cell_shift)
    return V_gsfe_delta(delta, a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum)



def W_phase_prime_raw(s, a=1.0, L=0.9, eps_lj=1.0, sigma=1.0, nsum=80, cell_shift=0.0):
    delta = phase_to_registry(s, a=a, cell_shift=cell_shift)
    return (a / (2.0 * np.pi)) * dV_gsfe_ddelta(delta, a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum)



def W_phase_unit_range(s, a=1.0, L=0.9, eps_lj=1.0, sigma=1.0, nsum=80, cell_shift=0.0):
    grid = np.linspace(0.0, a, 4000, endpoint=False)
    vals = V_gsfe_delta(grid, a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum)
    vmin = np.min(vals)
    vmax = np.max(vals)
    raw = W_phase_raw(s, a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum, cell_shift=cell_shift)
    return (raw - vmin) / (vmax - vmin)



def preferred_registry(a=1.0, L=0.9, eps_lj=1.0, sigma=1.0, nsum=80):
    e_AA = V_gsfe_delta(0.0, a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum)
    e_AB = V_gsfe_delta(0.5 * a, a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum)
    if e_AA < e_AB:
        pref = "AA (on-top)"
    elif e_AB < e_AA:
        pref = "AB (midpoint / in-between)"
    else:
        pref = "degenerate"
    return float(e_AA), float(e_AB), pref



def plot_one_choice(a=1.0, L=0.9, eps_lj=1.0, sigma=1.0, nsum=80, cell_shift=0.0, label=None):
    delta = np.linspace(0.0, a, 1200, endpoint=False)
    Vd = V_gsfe_delta(delta, a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum)

    s = np.linspace(-np.pi, np.pi, 1200, endpoint=False)
    Ws = W_phase_unit_range(s, a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum, cell_shift=cell_shift)

    e_AA, e_AB, pref = preferred_registry(a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum)
    title = label or f"a={a}, sigma={sigma}, eps_lj={eps_lj}, L={L}"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(delta, Vd, lw=2)
    axes[0].axvline(0.0, ls='--', alpha=0.6)
    axes[0].axvline(0.5 * a, ls='--', alpha=0.6)
    axes[0].set_xlabel(r"registry $\delta$")
    axes[0].set_ylabel(r"$V_{\mathrm{GSFE}}(\delta)$")
    axes[0].set_title("raw stacking energy")
    axes[0].grid(True, ls='--', alpha=0.4)
    axes[0].text(0.03 * a, np.min(Vd) + 0.1 * (np.max(Vd) - np.min(Vd)), f"AA: {e_AA:.4f}\nAB: {e_AB:.4f}\nmin: {pref}")

    axes[1].plot(s, Ws, lw=2)
    axes[1].set_xlabel(r"phase $s$")
    axes[1].set_ylabel(r"normalized $W(s)$")
    axes[1].set_title("2π-periodic phase potential")
    axes[1].grid(True, ls='--', alpha=0.4)

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes



def sweep_L_values(L_values, a=1.0, sigma=1.0, eps_lj=1.0, nsum=80):
    print("Sweep over L values")
    print("-------------------")
    for L in L_values:
        e_AA, e_AB, pref = preferred_registry(a=a, L=L, eps_lj=eps_lj, sigma=sigma, nsum=nsum)
        print(f"L={L:6.3f}   E_AA={e_AA:12.6f}   E_AB={e_AB:12.6f}   preferred: {pref}")


if __name__ == "__main__":
    # Example 1: AB-like minimum (on-top is too repulsive)
    plot_one_choice(a=1.0, sigma=1.0, eps_lj=1.0, L=0.90, nsum=100, label="AB-like example")

    # Example 2: AA-like minimum (larger layer distance makes on-top favorable)
    plot_one_choice(a=1.0, sigma=1.0, eps_lj=1.0, L=1.40, nsum=100, label="AA-like example")

    # Sweep L to see where the preference changes.
    sweep_L_values([0.8, 0.9, 1.0, 1.1, 1.2, 1.27, 1.35, 1.4], a=1.0, sigma=1.0, eps_lj=1.0, nsum=100)

    plt.show()
