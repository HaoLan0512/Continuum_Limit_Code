"""Compare constrained virtual-time relaxation with convergence diagnostics.

Method:
    Finite-difference derivative symbols applied by FFT with semi-implicit
    virtual time, configurable projections, and optional strong pinning.
Purpose:
    Compare baseline and custom solutions while recording physical/FFT
    residuals and finite-difference energy histories.
Inputs:
    Eta, grid and time-step settings, order, projection and initialization
    options, tolerance, and iteration limit.
Outputs:
    Console diagnostics and interactive solution/convergence figures; no files.
Output location:
    None.
Dependencies:
    NumPy and Matplotlib; no local module imports.
Related files:
    Extends the constraint comparison with diagnostics and strong pinning.

Modifications:
1. Records the residual of the PDE (real-space and FFT) each iteration.
2. Computes and records the energy each iteration.
3. Implements a strong pinning mode (`kind="pin_pi_ends"`) using U(x)=sin(x)*v(x).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional

# --------------------------------------------------------------------------- #
#                            admissible-set projection                        #
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
    """Project (U, f_val) onto a user-specified admissible set.

    Parameters
    ----------
    U, f_val
        Arrays to be modified (a *copy* is returned). Pass `None` for any quantity that should remain untouched.
    x
        Grid points on [0, 2π).
    mean_U, mean_f
        Enforce zero mean on U and/or f_val.
    pin_pi
        Impose U(π) = 0.
    pin_ends
        Impose periodic end-point pinning U(0) = U(2π) = 0.
    """
    if U is not None:
        U = U.copy()
        if mean_U:
            U -= U.mean()
        if pin_pi:
            idx_pi = int(np.round(
                len(x) * 0.5))  # index closest to x = π for uniform grid
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
#                            initialization helper                            #
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
    """Return an initial array U0 according to the chosen *kind*.

    Available kinds (case-insensitive)
    ----------------------------------
    "zero"       – all zeros (default)
    "constant"   – constant array with value `const_val`
    "odd" / "sin"  – sin(x)   (odd with respect to π)
    "even"/ "cos"  – cos(x)   (even with respect to π)
    "random"     – uniform random values in [a, b]
    """
    rng = np.random.default_rng(seed)
    kind = kind.lower()
    if kind == "zero":
        return np.zeros_like(x)
    elif kind == "constant":
        return np.full_like(x, const_val)
    elif kind in {"odd", "sin"}:
        return np.sin(x)  # scale down to avoid large initial values
    elif kind in {"even", "cos"}:
        return np.cos(x)  # scale down to avoid large initial values
    elif kind == "random":
        return rng.uniform(a, b, size=x.shape)
    else:
        raise ValueError(f"Unknown initialisation kind '{kind}'.")


# --------------------------------------------------------------------------- #
#                         finite-difference symbol (Laplacian)                #
# --------------------------------------------------------------------------- #


def fd_symbol(N: int, dx: float, fd_order: int = 2) -> np.ndarray:
    """
    Return λ_fd[k] = discrete symbol of -∂_xx on a uniform 2π-periodic grid.
    fd_order = 2 -> 4 sin^2(θ/2) / dx^2        (3-point stencil)
    fd_order = 4 -> (5/2 − (8/3)cosθ + (1/6)cos2θ) / dx^2   (5-point stencil)
    """
    k_phys = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    theta = k_phys * dx
    if fd_order == 2:
        lam = 4.0 * np.sin(theta / 2.0)**2 / dx**2  # λ_2(θ)
    elif fd_order == 4:
        lam = (2.5 - (8.0 / 3.0) * np.cos(theta) +
               (1.0 / 6.0) * np.cos(2.0 * theta)) / dx**2  # λ_4(θ)
    else:
        raise ValueError("fd_order must be 2 or 4")
    # lam[0] = 0.0  # ensure the discrete Laplacian annihilates constant modes (solvability condition)
    return lam


def fd_diff_symbol(N: int, dx: float, fd_order: int = 2) -> np.ndarray:
    """
    Discrete Fourier symbol of the *first* derivative operator
    matching the central finite–difference stencil used for the Laplacian.

      fd_order = 2  →  (U_{i+1} - U_{i-1}) / (2dx)
                    ⇒  multiplier  i·sinθ / dx

      fd_order = 4  →  ( -U_{i+2} + 8U_{i+1} - 8U_{i-1} + U_{i-2} ) / (12dx)
                    ⇒  multiplier  i·( 8 sinθ − sin2θ ) / (6dx)
    """
    k_phys = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    theta = k_phys * dx

    if fd_order == 2:
        mu = (1j * np.sin(theta)) / dx
    elif fd_order == 4:
        mu = 1j * (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / (6.0 * dx)
    else:
        raise ValueError("fd_order must be 2 or 4")
    return mu


# --------------------------------------------------------------------------- #
#                    semi-implicit Euler solver (virtual time)                #
# --------------------------------------------------------------------------- #


def solve_virtual_time(
    eta: float,
    *,
    N: int = 512,
    h: float = 0.4,
    fd_order: int = 4,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    admissible_opts: Optional[Dict[str, Any]] = None,
    init_opts: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, bool, int, np.ndarray, np.ndarray, np.ndarray]:
    """Perform semi-implicit Euler iterations: U_new = (I - h ∇^2)^{-1}[U + h f(U)], with optional constraints.
    
    Returns:
        U_final (np.ndarray): Solution array on [0,2π).
        converged (bool): True if convergence criterion met within max_iter.
        iter_count (int): Number of iterations performed.
        residuals_real (np.ndarray): Max residual |U_xx - f(U)| at each iteration (real-space).
        residuals_fft (np.ndarray): Max residual in Fourier domain at each iteration.
        energies (np.ndarray): Energy E[U] at each iteration.
    """
    admissible_opts = admissible_opts or {}
    init_opts = init_opts or {}

    L = 2 * np.pi
    x = np.linspace(0.0, L, N, endpoint=False)
    dx = x[1] - x[0]
    S = np.sin(
        x
    )  # sin(x) values for the grid, used in pinning via reparameterization

    lam = fd_symbol(N, dx, fd_order)  # symbol of -∇^2
    denom = 1.0 + h * lam  # diagonal of (I - h ∇^2) in Fourier (all > 0 except index 0 which is 1.0)

    # Prepare lists to record convergence metrics
    residuals_real = []
    residuals_fft = []
    energies = []

    # --- initial guess U (and v if using reparameterization) ---------------- #
    if admissible_opts.get("kind") == "pin_pi_ends":
        # Strong pinning mode: represent U = sin(x) * v(x)
        # Initialize U and then project it to satisfy U(0)=U(π)=U(2π)=0
        U_init = initialisation(x, **init_opts)
        # Impose odd symmetry about π: U_proj(x) = 0.5 * [U_init(x) - U_init(2π - x)]
        U_rev = U_init[::-1]  # U values reversed (approx U(2π - x) on grid)
        U_proj = 0.5 * (U_init - U_rev)
        U_proj[0] = 0.0  # enforce U(0)=0 explicitly
        if N % 2 == 0:
            U_proj[N //
                   2] = 0.0  # enforce U(π)=0 explicitly (if π is a grid point)
        U = U_proj
        # Compute initial v(x) = U(x) / sin(x) (well-defined except where sin(x)=0)
        v = np.empty_like(U)
        for i in range(N):
            if abs(S[i]) > 1e-12:
                v[i] = U[i] / S[i]
            else:
                # Define v at singular points by continuity (approximate using neighboring points)
                if i == 0:
                    v[i] = U[1] / S[1] if abs(S[1]) > 1e-12 else 0.0
                elif N % 2 == 0 and i == N // 2:
                    v[i] = U[i - 1] / S[i - 1] if abs(S[i -
                                                        1]) > 1e-12 else 0.0
                else:
                    v[i] = 0.0
        # Enforce zero-mean on U if requested (though U as sin-series should already have mean ~ 0)
        if admissible_opts.get("mean_U"):
            U -= U.mean()
    else:
        # Standard initialization (no strong pinning)
        U = initialisation(x, **init_opts)
        # Apply any initial admissible-set projections (e.g., mean zero, simple pinning)
        U, _ = admissible_set(U, None, x, **admissible_opts)
        v = None  # not used in standard mode

    # --- iteration loop (virtual time stepping) ---------------------------- #
    for k in range(max_iter):
        U_old = U.copy()

        if admissible_opts.get("kind") == "pin_pi_ends":
            # Recompute U from current v to ensure U stays in pinned subspace
            U = S * v  # U(x) = sin(x)*v(x), so U(0)=U(π)=U(2π)=0 automatically
            if admissible_opts.get("mean_U"):
                U -= U.mean()  # maintain zero mean if required
        else:
            # Enforce standard admissible-set constraints on U (mean, simple pinning) before evaluating f
            U, _ = admissible_set(U, None, x, **admissible_opts)

        # Evaluate the nonlinear term f(U).
        # f(U) = -(√2 * η^2 / π) * sin(x + 2π√2 * U(x)).
        f_val = -(np.sqrt(2) * eta**2 / np.pi) * np.sin(x + 2 * np.pi *
                                                        np.sqrt(2) * U)
        # Apply admissible-set projection to f if needed (e.g., enforce zero mean on f)
        if admissible_opts.get("kind") == "pin_pi_ends":
            if admissible_opts.get("mean_f"):
                f_val -= f_val.mean()
        else:
            _, f_val = admissible_set(None, f_val, x, **admissible_opts)

        # Form the RHS for the linear solve: R = U + h * f(U)
        R = U + h * f_val
        # Enforce solvability (zero mean) if neither U nor f have a mean-zero constraint.
        # This following prevents accumulation of any constant mode (which ∇^2 cannot eliminate). Turn it on if needed.

        # if not (admissible_opts.get("mean_U")
        #         or admissible_opts.get("mean_f")):
        #     R -= R.mean()

        # Solve (I - h Δ) U_new = R in Fourier space (Δ = Laplacian)
        R_hat = np.fft.fft(R)
        U_hat_new = R_hat / denom  # safe since denom[0]=1 (lam[0]=0)
        U_temp = np.real(np.fft.ifft(U_hat_new))

        if admissible_opts.get("kind") == "pin_pi_ends":
            # Project U_temp onto the sin(x)*v subspace (odd about π)
            U_rev = U_temp[::-1]
            U_proj = 0.5 * (U_temp - U_rev)
            U_proj[0] = 0.0
            if N % 2 == 0:
                U_proj[N // 2] = 0.0
            U_new = U_proj
            # Update v for next iteration: v = U_new / sin(x)
            new_v = np.empty_like(U_new)
            for i in range(N):
                if abs(S[i]) > 1e-12:
                    new_v[i] = U_new[i] / S[i]
                else:
                    if i == 0:
                        new_v[i] = U_new[1] / S[1] if abs(
                            S[1]) > 1e-12 else 0.0
                    elif N % 2 == 0 and i == N // 2:
                        new_v[i] = U_new[i - 1] / S[i - 1] if abs(
                            S[i - 1]) > 1e-12 else 0.0
                    else:
                        new_v[i] = 0.0
            v = new_v
            U = U_new
        else:
            # Apply final admissible-set projection on U (ensures constraints exactly)
            U_new, _ = admissible_set(U_temp, None, x, **admissible_opts)
            U = U_new

        # --- Compute residual error to check convergence --- #
        # f_exact = (√2 * η^2 / π) * sin(x + 2π√2 * U) is the right-hand side of the Euler–Lagrange equation (with positive sign).
        f_exact = -(f_val)
        # Compute U_xx via spectral differentiation (using lam, the symbol for -∂xx)
        U_hat = np.fft.fft(U)
        U_xx_hat = -lam * U_hat  # -lam * U_hat corresponds to +∂^2 U in physical space
        U_xx = np.real(np.fft.ifft(U_xx_hat))
        # Calculate residual in real space and Fourier space
        residual_phys = U_xx - f_exact
        max_residual = np.max(np.abs(residual_phys))
        idx_res_real = int(np.argmax(np.abs(residual_phys)))
        residuals_real.append(max_residual)
        # Fourier-domain residual: ideally -lam*U_hat == F_exact_hat
        F_exact_hat = np.fft.fft(f_exact)
        residual_freq = (-lam * U_hat) - F_exact_hat
        max_residual_fft = np.max(np.abs(residual_freq))
        idx_res_fft = int(np.argmax(np.abs(residual_freq)))
        residuals_fft.append(max_residual_fft)

        # ---- energy using *finite-difference* derivative --------------------------
        mu = fd_diff_symbol(N, dx,
                            fd_order)  # << use same fd_order as the Laplacian
        U1_hat = mu * U_hat  # Fourier coeffs of U′  (discrete FD)
        U1 = np.real(np.fft.ifft(U1_hat))  # back to physical space

        energy = (np.pi / eta**2) * np.sum(U1**2) * dx - (
            1.0 / np.pi) * np.sum(np.cos(x + 2 * np.pi * np.sqrt(2) * U)) * dx
        energies.append(energy)

        # Check convergence: stop if the maximum residual is below tolerance
        if max_residual < tol:
            return U, True, (k + 1), np.array(residuals_real), np.array(
                residuals_fft), np.array(energies), idx_res_real, idx_res_fft
        # (We could also check relative change in U, but residual criteria is more direct for PDE convergence.)

    # If reached max_iter without satisfying tolerance
    return U, False, max_iter, np.array(residuals_real), np.array(
        residuals_fft), np.array(energies), idx_res_real, idx_res_fft


# --------------------------------------------------------------------------- #
#                            helpers for legend text                          #
# --------------------------------------------------------------------------- #


def describe_admissible(opts: Dict[str, Any]) -> str:
    labels = []
    if opts.get("mean_U"):
        labels.append("⟨U⟩=0")
    if opts.get("mean_f"):
        labels.append("⟨f⟩=0")
    kind = opts.get("kind", None)
    if kind == "pin_pi_ends":
        labels.append("U(0,π,2π)=0 via sin")
    else:
        if opts.get("pin_pi"):
            labels.append("U(π)=0")
        if opts.get("pin_ends"):
            labels.append("U(0,2π)=0")
    return ", ".join(labels) if labels else "none"


def describe_init(opts: Dict[str, Any]) -> str:
    kind = opts.get("kind", "zero").lower()
    if kind == "constant":
        val = opts.get("const_val", None)
        return f"constant({val})"
    elif kind == "random":
        a = opts.get("a", None)
        b = opts.get("b", None)
        return f"random[{a},{b}]"
    else:
        return kind


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #


def main() -> None:
    """
    Example usage and comparison between baseline and custom runs.
    Available initialisation kinds: "zero", "constant", "odd"/"sin", "even"/"cos", "random".
    """
    # --- physical & numerical parameters --- #
    eta = 10
    N = 512
    h = 0.003
    fd_order = 4

    # --- custom run with strong pinning (U(0)=U(π)=U(2π)=0) --- #
    custom_adm: Dict[str, Any] = {
        # "kind": "pin_pi_ends",
        "mean_U": False,
        "mean_f": False
    }
    custom_init: Dict[str, Any] = {
        "kind": "sin",  # try a random initial guess
        "const_val": 1 / np.sqrt(2),  # (not used for random)
        "a": -1 / (1.5 * np.sqrt(2)),
        "b": 1 / (1.5 * np.sqrt(2)),
        "seed": 3  # for reproducibility of the random initial condition
    }

    # --- baseline run (old behavior: only enforce mean zero) --- #
    base_adm: Dict[str, Any] = {
        "mean_U": True,
        "mean_f": True
        # no pinning; solution will float but mean zero is enforced
    }
    base_init: Dict[str, Any] = {"kind": "zero"}

    # --- solve both scenarios --- #
    U_base, ok_base, nit_base, res_base, resfft_base, E_base, idxp_b, idxf_b = solve_virtual_time(
        eta,
        N=N,
        h=h,
        fd_order=fd_order,
        admissible_opts=base_adm,
        init_opts=base_init)
    U_cust, ok_cust, nit_cust, res_cust, resfft_cust, E_cust, idxp_c, idxf_c = solve_virtual_time(
        eta,
        N=N,
        h=h,
        fd_order=fd_order,
        admissible_opts=custom_adm,
        init_opts=custom_init)

    if not ok_base or not ok_cust:
        print("WARNING – convergence not achieved in some run (base OK?:",
              ok_base, ", custom OK?:", ok_cust, ")")

    if not ok_base:
        x_fail = 2 * np.pi * idxp_b / N
        print(
            f"[baseline]  max-residual at grid index {idxp_b}  (x ≈ {x_fail:.4f})"
        )
        k_fail = 2 * np.pi * idxf_b / N
        print(
            f"[baseline]  max-FFT-residual at k-index {idxf_b}  (|k| ≈ {k_fail:.4f})"
        )

    if not ok_cust:
        x_fail = 2 * np.pi * idxp_c / N
        print(
            f"[custom]    max-residual at grid index {idxp_c}  (x ≈ {x_fail:.4f})"
        )
        k_fail = 2 * np.pi * idxf_c / N
        print(
            f"[custom]    max-FFT-residual at k-index {idxf_c}  (|k| ≈ {k_fail:.4f})"
        )

    # Compare final energies to see if custom solution is a different (likely local) minimizer
    final_E_base = E_base[-1] if len(E_base) > 0 else None
    final_E_cust = E_cust[-1] if len(E_cust) > 0 else None

    if final_E_base is not None and final_E_cust is not None:
        # round to two decimals
        Eb2 = round(final_E_base, 2)
        Ec2 = round(final_E_cust, 2)

        print(f"Final energy (baseline) ≈ {Eb2:.2f},  (custom) ≈ {Ec2:.2f}")
        if Eb2 == Ec2:
            print("→  They have the *same* energy (up to 2 d.p.).")
        elif Ec2 < Eb2:
            print("→  Custom solution has lower energy than baseline.")
        else:
            print(
                "→  Baseline solution has lower (or equal) energy compared to custom."
            )

    # --- Plot the solution profiles U(x) --- #
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
    plt.title(rf"η = {eta}, FD order = {fd_order}")
    plt.xlabel("x")
    plt.ylabel("U(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- (Optional) Plot residual and energy convergence curves --- #
    plt.figure()
    plt.semilogy(res_base, label="Residual (baseline)")
    plt.semilogy(res_cust, "--", label="Residual (custom)")
    plt.xlabel("Iteration")
    plt.ylabel("Max |U_xx - f(U)|")
    plt.title("Residual Error vs. Iteration")
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogy(resfft_base, label="FFT residual (baseline)")
    plt.semilogy(resfft_cust, "--", label="FFT residual (custom)")
    plt.xlabel("Iteration")
    plt.ylabel("Max | –λ Û  – F_exact_hat |")
    plt.title("FFT-Domain Residual vs. Iteration")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(E_base, label="Energy (baseline)")
    plt.plot(E_cust, "--", label="Energy (custom)")
    plt.xlabel("Iteration")
    plt.ylabel("E[U]")
    plt.title("Energy Functional vs. Iteration")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
