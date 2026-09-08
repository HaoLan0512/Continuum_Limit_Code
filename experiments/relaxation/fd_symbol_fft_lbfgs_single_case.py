"""Compute one L-BFGS-B minimizer using FFT-applied finite-difference symbols.

Method:
    Centered three-point derivative symbols evaluated through FFT and L-BFGS-B.
Purpose:
    Minimize the periodic continuum energy and save the displacement and plots.
Inputs:
    In-file values N=256, eps=0.1, and DEALIAS=False.
Outputs:
    u_min_fft.npy, a minimizer PDF, and a modified-disregistry PDF.
Output location:
    outputs/relaxation/output_fd_symbol_fft_lbfgs_single_case.
Dependencies:
    NumPy, SciPy, and Matplotlib; no local module imports.
Related files:
    Produces the input used by the FFT/FD-symbol Bloch band study.

Implementation notes:
    The minimized energy is

    I(u) = ∫₀^{2π} ½ ε² (u′)² − cos(x+u) dx

on the periodic interval [0, 2π] using L‑BFGS‑B.
The derivatives are evaluated with centered finite-difference symbols via FFT,
while the nonlinear cosine term is treated point-wise in
physical space.

The historical ``DEALIAS`` flag remains false; the active derivative function
uses the finite-difference symbols without the commented de-aliasing branch.
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from pathlib import Path

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
N = 256  # number of grid points on [0, 2π)
eps = 0.1  # ε in the energy
DEALIAS = False  # True → zero highest 1/3 of Fourier modes
SAVE_PATH = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "relaxation"
    / "output_fd_symbol_fft_lbfgs_single_case"
)

# --------------------------------------------------------------------------- #
# Spectral differentiation (with optional 2/3‑rule dealiasing)
# --------------------------------------------------------------------------- #
# def spectral_derivatives(u: np.ndarray, dx: float, dealias: bool = False):
#     """
#     Returns (u_x, u_xx) evaluated via FFT on a uniform periodic grid.
#     If `dealias` is True, applies the 2/3‑rule before differentiation.
#     """
#     N = u.size
#     # Fourier wavenumbers: 0, 1, …, N/2, −N/2+1, …, −1  (scaled for 2π‑periodicity)
#     k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)  # shape (N,)

#     u_hat = np.fft.fft(u)

#     if dealias:
#         # Zero out modes |k| > (N/3)*(2π/L)
#         cut = N // 3
#         u_hat[cut:N - cut] = 0.0

#     # First derivative:  u_x ↔ i k û
#     u_x = np.fft.ifft(1j * k * u_hat).real

#     # Second derivative: u_xx ↔ −k² û
#     u_xx = np.fft.ifft(-(k**2) * u_hat).real

#     return u_x, u_xx


def spectral_derivatives(u: np.ndarray, dx: float):
    """
    Finite-difference derivatives evaluated via FFT on a uniform periodic grid.

    Returns
    -------
    u_x  : np.ndarray
        First derivative using the centred 3-point stencil (O(dx^2)).
    u_xx : np.ndarray
        Second derivative (Laplacian) using the same stencil.
    """
    N = u.size

    # Fourier wavenumbers in radians per unit length
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)  # shape (N,)
    theta = k * dx  # dimensionless 2πk/N

    # FD Fourier symbols
    D1 = 1j * np.sin(theta) / dx  # i·sin(k dx)/dx
    D2 = -4.0 * np.sin(theta / 2.0)**2 / dx**2  # -4 sin²(k dx/2)/dx²

    u_hat = np.fft.fft(u)

    u_x = np.fft.ifft(D1 * u_hat).real
    u_xx = np.fft.ifft(D2 * u_hat).real

    return u_x, u_xx


# --------------------------------------------------------------------------- #
# Energy functional I(u) and its gradient
# --------------------------------------------------------------------------- #
def energy(u: np.ndarray, x: np.ndarray, dx: float, eps: float):
    """
    Computes the energy I(u).  Uses spectral derivative for (u′)² term
    and point‑wise evaluation for the cosine term.
    """
    u_x, _ = spectral_derivatives(u, dx)
    integrand = 0.5 * eps**2 * u_x**2 - np.cos(x + u)
    return np.sum(integrand) * dx


def grad_energy(u: np.ndarray, x: np.ndarray, dx: float, eps: float):
    """
    Computes the gradient ∂I/∂u evaluated at the grid points.
    For periodic BC the variational derivative is −ε² u_xx + sin(x+u).
    """
    _, u_xx = spectral_derivatives(u, dx)
    grad = (-eps**2 * u_xx + np.sin(x + u)) * dx
    return grad


# --------------------------------------------------------------------------- #
# Wrapper for scipy optimiser
# --------------------------------------------------------------------------- #
def make_energy_and_grad(x: np.ndarray, dx: float, eps: float):
    """Returns two callables f(u) and g(u) for L‑BFGS‑B."""
    def _f(u):
        return energy(u, x, dx, eps)

    def _g(u):
        return grad_energy(u, x, dx, eps)

    return _f, _g


# --------------------------------------------------------------------------- #
# Main minimisation routine
# --------------------------------------------------------------------------- #
def compute_minimiser(N: int, eps: float):
    dx = 2.0 * np.pi / N
    x = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)

    f, g = make_energy_and_grad(x, dx, eps)
    u0 = np.zeros(N)  # initial guess

    u_opt, f_opt, _info = fmin_l_bfgs_b(f, u0, fprime=g, pgtol=1e-12)
    return x, dx, u_opt, f_opt


# --------------------------------------------------------------------------- #
# Plot wrapper
# --------------------------------------------------------------------------- #
def plot_solution(x, u, eps, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # u(x)
    plt.figure()
    plt.plot(x, u, lw=1.2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x)$")
    plt.title(rf"Minimiser $u$  ($\varepsilon={eps}$)")
    plt.tight_layout()
    plt.savefig(out_dir / f"minimiser_eps_{eps:.3g}.pdf")

    # x + u(x)
    plt.figure()
    plt.plot(x, x + u, lw=1.2)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$x+u(x)$")
    plt.title(rf"Modified disregistry $x+u$  ($\varepsilon={eps}$)")
    plt.tight_layout()
    plt.savefig(out_dir / f"disregistry_eps_{eps:.3g}.pdf")


# --------------------------------------------------------------------------- #
# Script
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    x, dx, u_min, E_min = compute_minimiser(N, eps)

    # Save minimiser to disk
    np.save(SAVE_PATH / "u_min_fft.npy", u_min)

    # Quick summary
    print(f"Finished minimisation with N={N}, ε={eps}")
    print(f"  Energy I(u) ≈ {E_min:.12e}")

    # Visualise
    plot_solution(x, u_min, eps, SAVE_PATH)

    # Show plots
    plt.show()
