"""Numerical solver for the 1D moiré Euler–Lagrange equation with
flexible admissible‐set projection and initialisation routines.

The script adds two utility functions that the user can tweak:

* ``admissible_set``   – impose mean‑zero / pinning conditions on ``U`` or ``f``.
* ``initialisation``   – generate the starting iterate ``U_0``.

A baseline run (old behaviour: ⟨U⟩=0 with zero initial data) is shown
alongside the custom run for direct comparison.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional

# --------------------------------------------------------------------------- #
#                            admissible‐set projection                        #
# --------------------------------------------------------------------------- #


def admissible_set(
    U: Optional[np.ndarray],
    f_val: Optional[np.ndarray],
    x: np.ndarray,
    *,
    mean_U: bool = False,
    mean_f: bool = False,
    pin_pi: bool = False,
    pin_ends: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Project *(U, f_val)* onto a user‑specified admissible set.

    Parameters
    ----------
    U, f_val
        Arrays to be modified (a *copy* is returned).  Pass ``None`` if the
        quantity should be left untouched.
    x
        Grid points on *[0,2π)*.
    mean_U, mean_f
        Enforce zero mean on *U* and/or *f_val*.
    pin_pi
        Impose ``U(π) = 0``.
    pin_ends
        Impose periodic end‑point pinning ``U(0) = U(2π) = 0``.
    """
    if U is not None:
        U = U.copy()
        if mean_U:
            U -= U.mean()
        if pin_pi:
            idx_pi = int(np.round(len(x) * 0.5))  # x≈π for uniform grid
            U[idx_pi] = 0.0
        if pin_ends:
            U[0] = 0.0
            U[-1] = 0.0

    if f_val is not None:
        f_val = f_val.copy()
        if mean_f:
            f_val -= f_val.mean()

    return U, f_val


# --------------------------------------------------------------------------- #
#                            initialisation helper                            #
# --------------------------------------------------------------------------- #


def initialisation(
    x: np.ndarray,
    *,
    kind: str = "zero",
    const_val: float = 0.0,
    a: float = 0.0,
    b: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return an initial array ``U_0`` matching the chosen *kind*.

    Available kinds (case‑insensitive)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ``"zero"``       – all zeros (default)
    ``"constant"``   – constant array with value ``const_val``
    ``"odd"`` / ``"sin"``  – *sin(x)* (odd wrt π)
    ``"even"``/ ``"cos"``  – *cos(x)* (even wrt π)
    ``"random"``     – uniform random values in ``[a,b]``
    """
    rng = np.random.default_rng(seed)
    kind = kind.lower()
    if kind == "zero":
        return np.zeros_like(x)
    elif kind == "constant":
        return np.full_like(x, const_val)
    elif kind in {"odd", "sin"}:
        return np.sin(x)
    elif kind in {"even", "cos"}:
        return np.cos(x)
    elif kind == "random":
        return rng.uniform(
            a, b, size=x.shape)  # a + (b - a) * np.random.rand(*x.shape)

    else:
        raise ValueError(f"Unknown initialisation kind '{kind}'.")


# --------------------------------------------------------------------------- #
#                            finite‑difference symbol                         #
# --------------------------------------------------------------------------- #


def fd_symbol(N: int, dx: float, fd_order: int = 2) -> np.ndarray:
    """
    Return λ_fd[k]  =  discrete symbol of  -∂xx on a uniform 2π-periodic grid.
    fd_order = 2 -> 4 sin²(θ/2)/dx²   (3-point)
    fd_order = 4 -> (5/2 − 8/3 cosθ + 1/6 cos2θ)/dx²   (5-point)
    """
    k_phys = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    theta = k_phys * dx
    if fd_order == 2:
        lam = 4.0 * np.sin(theta / 2.0)**2 / dx**2  # λ₂(θ)
    elif fd_order == 4:
        lam = (2.5 - (8.0 / 3.0) * np.cos(theta) +
               (1.0 / 6.0) * np.cos(2.0 * theta)) / dx**2  # λ₄(θ)
    else:
        raise ValueError("fd_order must be 2 or 4")
    lam[0] = 0.0  #ensure the discrete Laplacian kills the constant grid vector.
    return lam


# --------------------------------------------------------------------------- #
#                         semi‑implicit Euler (virtual time)                  #
# --------------------------------------------------------------------------- #


def solve_virtual_time(
    eta: float,
    *,
    N: int = 512,
    h: float = 0.4,
    fd_order: int = 4,
    tol: float = 1e-8,
    max_iter: int = 100_000,
    admissible_opts: Optional[Dict[str, Any]] = None,
    init_opts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, bool, int]:
    """Semi‑implicit Euler:  ``(I - h∇²)^{-1}(U + h f(U))`` with constraints."""
    admissible_opts = admissible_opts or {}
    init_opts = init_opts or {}

    L = 2 * np.pi
    x = np.linspace(0.0, L, N, endpoint=False)
    dx = x[1] - x[0]

    lam = fd_symbol(N, dx, fd_order)
    denom = 1.0 + h * lam  # (I - h∇²) = (I + h(-Δ)) = (I + h λ_fd) in Fourier

    # --- initial U --------------------------------------------------------- #
    U = initialisation(x, **init_opts)
    print(U)

    for k in range(max_iter):
        U_old = U.copy()

        # Enforce projection on U before evaluating f(U)
        U, _ = admissible_set(U, None, x, **admissible_opts)

        f_val = -(np.sqrt(2) * eta**2 / np.pi) * np.sin(x + 2 * np.pi *
                                                        np.sqrt(2) * U)
        # Optional projection on f(U)
        _, f_val = admissible_set(None, f_val, x, **admissible_opts)

        R = U + h * f_val  # right‑hand side in physical space
        # R -= R.mean()  # keep solvability condition ⟨R⟩=0
        R_hat = np.fft.fft(R)
        U_hat = np.zeros_like(R_hat)
        nz_idx = lam >= 0.0  # include index 0
        U_hat[nz_idx] = R_hat[nz_idx] / denom[nz_idx]
        U = np.real(np.fft.ifft(U_hat))

        # Final projection on U to respect constraints exactly
        U, _ = admissible_set(U, None, x, **admissible_opts)

        relerr = np.linalg.norm(U - U_old) / max(1e-30, np.linalg.norm(U_old))
        if relerr < tol:
            return U, True, k + 1
    return U, False, max_iter


# --------------------------------------------------------------------------- #
#                            helpers for legend text                          #
# --------------------------------------------------------------------------- #


def describe_admissible(opts: Dict[str, Any]) -> str:
    labels = []
    if opts.get("mean_U"):
        labels.append("⟨U⟩=0")
    if opts.get("mean_f"):
        labels.append("⟨f⟩=0")
    if opts.get("pin_pi"):
        labels.append("U(π)=0")
    if opts.get("pin_ends"):
        labels.append("U(0,2π)=0")
    return ", ".join(labels) if labels else "none"


def describe_init(opts: Dict[str, Any]) -> str:
    kind = opts.get("kind", "zero").lower()
    if kind == "constant":
        const_val = opts.get('const_val', None)
        return f"constant({const_val})"
    elif kind == "random":
        a = opts.get('a', None)
        b = opts.get('b', None)
        return f"random[{a},{b}]"
    else:
        return kind


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #


def main() -> None:
    """
    Available kinds (case‑insensitive)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ``"zero"``       – all zeros (default)
    ``"constant"``   – constant array with value ``const_val``
    ``"odd"`` / ``"sin"``  – *sin(x)* (odd wrt π)
    ``"even"``/ ``"cos"``  – *cos(x)* (even wrt π)
    ``"random"``     – uniform random values in ``[a,b]``
    """
    # --------------------------------------------------------------------------- #
    #                            parameters table                                 #
    # --------------------------------------------------------------------------- #
    # eta  N      h     fd_order
    # 1    512   0.4      4
    # 1.3  512   0.2      4
    # 3    512  0.035     4
    # 10   512  0.003     4
    # --------------------------------------------------------------------------- #
    # --- physical & numerical parameters ----------------------------------- #
    eta = 10.0
    N = 512
    h = 0.003
    fd_order = 4

    # --- custom run setup -------------------------------------------------- #
    custom_adm: Dict[str, Any] = dict(mean_U=False,
                                      mean_f=False,
                                      pin_pi=True,
                                      pin_ends=True)
    custom_init: Dict[str, Any] = dict(
        kind="random",  # choose from "zero", "constant", "odd", "even", "random"
        # "constant" or "random" requires additional parameters below:
        const_val=np.pi,
        a=0.0,
        b=3.0,
        seed=3,  # for reproducibility
    )

    # --- baseline (old behaviour) ------------------------------------------ #
    base_adm: Dict[str, Any] = dict(mean_U=True, mean_f=True)
    base_init: Dict[str, Any] = dict(kind="zero")

    # --- solver calls ------------------------------------------------------ #
    U_base, ok_base, nit_base = solve_virtual_time(
        eta,
        N=N,
        h=h,
        fd_order=fd_order,
        admissible_opts=base_adm,
        init_opts=base_init,
    )

    U_cust, ok_cust, nit_cust = solve_virtual_time(
        eta,
        N=N,
        h=h,
        fd_order=fd_order,
        admissible_opts=custom_adm,
        init_opts=custom_init,
    )

    if not (ok_base and ok_cust):
        print("WARNING – at least one run did not converge (base, custom):",
              ok_base, ok_cust)

    # --- plotting ---------------------------------------------------------- #
    x = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

    plt.figure()
    plt.plot(
        x,
        U_base,
        label=
        f"baseline | {describe_admissible(base_adm)}, init={describe_init(base_init)}"
    )
    plt.plot(
        x,
        U_cust,
        "--",
        label=
        f"custom | {describe_admissible(custom_adm)}, init={describe_init(custom_init)}"
    )

    plt.title(rf"η={eta}, FD order={fd_order}")
    plt.xlabel("X")
    plt.ylabel("U(X)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
