"""
Spectral solver for the 1‑D Bloch eigen‑problem

    H ψ(x) = E ψ(x),   H = −(ε²/2) d²/dx² + V(x),   ψ(x+L)=e^{ikL} ψ(x),

on a uniform period‑cell grid  x_j = x0 + j·Δx,  j = 0,…,N−1. 

If you only need the periodic envelope p(x,k) (i.e. without the e^{ikx} phase)
pass ``return_envelope=True``.
"""
# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
# --------------------------------------------------------------------------- #
# __all__ defines the public API of this module; only 'bloch' will be imported.
__all__ = ["bloch"]


def _fix_global_phase(psi: np.ndarray) -> np.ndarray:
    """Rotate each column so that its first non‑zero entry is real and positive."""
    out = psi.copy()
    for n in range(out.shape[1]):
        col = out[:, n]
        # find first component with magnitude > tol
        for c in col:
            if abs(c) > 1e-14:
                phase = c / abs(c)
                out[:, n] /= phase
                break
    return out


def bloch(
    V: ArrayLike,
    x: ArrayLike,
    eps: float,
    k: float,
    *,
    fd_order: int = 4,  # ← 2  or  4
    return_envelope: bool = True,
):
    """
    Bloch eigen-problem solver using an FFT plane-wave basis.

    H ψ = E ψ,   H = −(ε²/2) d²/dx² + V(x),   ψ(x+L)=e^{ikL} ψ(x)

    The kinetic term −(ε²/2)∂ₓₓ is discretised by the finite-difference setting
    evaluated in Fourier space:

        fd_order = 2  →  3-point (2-nd-order)   λ₂(θ) = 4 sin²(θ/2) / Δx²
        fd_order = 4  →  5-point (4-th-order)   λ₄(θ) = (5/2 − 8/3 cosθ + 1/6 cos2θ) / Δx²
        θ = (G + k) Δx,  G = 2π·fftfreq(N, d=Δx)

    Parameters
    ----------
    V
        Potential sampled at the **uniform** grid ``x`` (shape (N,)).
    x
        Uniform 1‑D grid covering exactly one period.
    eps
        ε in −(ε²/2) Δ.
    k
        Bloch wave‑number (same units as 2π/L).  For the first Brillouin zone
        use k∈[−π/L, π/L].
    fd_order : {2, 4}, optional
        Order of the finite-difference Laplacian (default 2).
    return_envelope : bool, optional
        If True return the periodic envelopes p(x,k) instead of the full ψ.

    Returns
    -------
    eigvals : ndarray, shape (N,)
        Sorted band energies.
    psi_x   : ndarray, shape (N, N)
        Columns are Bloch functions ψ(x) (unit 2‑norm, real first entry).
    p_x     : ndarray, shape (N, N), *optional*
        Periodic envelopes p(x,k) with the same phase convention.
    """
    # ---- convert & basic geometry ----------------------------------- #
    V = np.asarray(V, dtype=float)
    x = np.asarray(x, dtype=float)
    if V.ndim != 1 or x.ndim != 1 or V.size != x.size:
        raise ValueError("V and x must be 1-D arrays of the same length")

    N = V.size
    dx = x[1] - x[0]

    # ---- plane-wave indices & kinetic diagonal ---------------------- #
    G = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    theta = (G + k) * dx  # (G+k)Δx

    if fd_order == 2:
        lap_fd = 4.0 * np.sin(theta / 2.0)**2 / dx**2  # λ₂
    elif fd_order == 4:
        lap_fd = (2.5 - (8.0 / 3.0) * np.cos(theta) +
                  (1.0 / 6.0) * np.cos(2.0 * theta)) / dx**2  # λ₄
    else:
        raise ValueError("fd_order must be 2 or 4")

    T_diag = 0.5 * eps**2 * lap_fd

    # ---- potential convolution matrix ------------------------------ #
    V_hat = np.fft.fft(V) / N
    idx = (np.subtract.outer(np.arange(N), np.arange(N))) % N
    H = V_hat[idx]
    np.fill_diagonal(H, H.diagonal() + T_diag)

    # ---- diagonalise ----------------------------------------------- #
    eigvals, coeffs = np.linalg.eigh(H)
    order = np.argsort(eigvals)
    eigvals = eigvals[order].real
    coeffs = coeffs[:, order]

    # ---- back to real space ---------------------------------------- #
    p_x = np.fft.ifft(coeffs, axis=0)
    phase = np.exp(1j * k * x)[:, None]
    psi_x = phase * p_x
    psi_x /= np.linalg.norm(psi_x, axis=0)
    psi_x = _fix_global_phase(psi_x)

    if return_envelope:
        return eigvals, psi_x / phase
    else:
        return eigvals, psi_x


# --------------------------------------------------------------------------- #
# Example usage / sanity check (run as script)                               #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 256
    x_grid = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    eps = 0.1
    k_val = 0.3
    V_grid = 0.5 * np.cos(x_grid)

    from bloch import bloch as bloch_fd  # finite‑difference reference

    E_fd, psi_fd = bloch_fd(V_grid, x_grid, eps, k_val)
    E_pw, psi_pw = bloch(V_grid, x_grid, eps, k_val)

    # print comparison ----------------------------------------------------- #
    print("   band        E_fd            E_fft        |ΔE|")
    for n in range(5):
        dE = abs(E_fd[n] - E_pw[n])
        print(f"    {n:2d}   {E_fd[n]: .8f}   {E_pw[n]: .8f}   {dE:.2e}")

    # quick visual check --------------------------------------------------- #
    plt.figure()
    plt.title("ψ₀(x): FD vs FFT (real part)")
    plt.plot(x_grid, psi_fd[:, 0].real, label="FD", lw=1.5)
    plt.plot(x_grid, psi_pw[:, 0].real, "--", label="FFT", lw=1.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
