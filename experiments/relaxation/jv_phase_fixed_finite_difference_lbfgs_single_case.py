"""Compute the phase-fixed minimizer of Equation (6) in the paper.

Method:
    Forward periodic finite differences, odd-variable reduction, deterministic
    multistart, and L-BFGS-B optimization.
Purpose:
    Use Theorem 1 to minimize J[v] in the odd, phase-fixed subspace, with
    v=x+u and u periodic.
Inputs:
    N=400, elastic coefficient alpha=1, and the canonical choice
    Phi(s)=cos(s), so W(s)=2*Phi(s).
Outputs:
    Phase-fixed u and v arrays, PNG profiles, convergence diagnostics, and
    a comparison of several initial guesses.
Output location:
    outputs/relaxation/output_jv_phase_fixed_finite_difference_lbfgs_single_case.
Dependencies:
    NumPy, SciPy, Matplotlib, and clr.relaxation.odd_periodic.
Related files:
    The unrestricted J[v] experiment compares with the same shared odd
    parametrization.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from clr.relaxation.odd_periodic import (
    build_odd_periodic as build_u,
    reduce_odd_gradient as reduced_gradient,
)

N = 400  # must be even
assert N % 2 == 0

# Equation (6): J[v] = integral (1/2)|v'|^2 + W(v), with W = 2*Phi.
alpha = 1.0
PGRAD_TOL = 5e-7  # Tolerance for the infinity norm of the reduced gradient, tells L-BFGS-B when the reduced optimization is sufficiently stationary
EL_RESIDUAL_TOL = 2e-5  # Tolerance for the infinity norm of the Euler-Lagrange residual, confirms that the reconstructed profile satisfies the full discrete Euler–Lagrange equation.

mid = N // 2
x = np.linspace(-np.pi, np.pi, N, endpoint=False)
dx = 2 * np.pi / N


def W(s):
    """Return W(s)=2*cos(s) for the paper's canonical Phi(s)=cos(s)."""
    return 2.0 * np.cos(s)


def Wprime(s):
    """Return the derivative of the canonical misfit potential W."""
    return -2.0 * np.sin(s)


def energy_full(u):
    """Evaluate the transformed discrete energy E[u] on the full grid."""
    du = np.roll(u, -1) - u
    elasticity = 0.5 * alpha * (du / dx)**2
    stacking = W(x + u)
    return np.sum(elasticity + stacking) * dx


def grad_full(u):
    """Return the gradient of E[u] with respect to all N grid values."""
    grad_elasticity = (alpha * (2 * u - np.roll(u, 1) - np.roll(u, -1)) / dx)
    grad_stacking = Wprime(x + u) * dx
    return grad_elasticity + grad_stacking


def energy_reduced(a):
    """Evaluate F(a)=E[build_u(a)] in the independent odd variables."""
    return energy_full(build_u(a))


def grad_reduced(a):
    """Return the chain-rule gradient of F(a)=E[build_u(a)]."""
    return reduced_gradient(grad_full(build_u(a)))


def initial_guesses():
    """Return deterministic zero, sinusoidal, and random multistart guesses."""
    rng = np.random.default_rng(0)
    x_free = x[1:mid]
    return {
        "zero": np.zeros(mid - 1),
        "sine": 0.5 * np.sin(x_free),
        "negative_sine": -0.5 * np.sin(x_free),
        "random": 0.1 * rng.standard_normal(mid - 1),
    }


def solve_from_initial_guess(a0):
    """Run L-BFGS-B once and require the requested gradient tolerance."""
    a_min, E_min, info = opt.fmin_l_bfgs_b(
        energy_reduced,
        a0,
        fprime=grad_reduced,
        pgtol=PGRAD_TOL,
        factr=10.0,
        maxiter=5000,
        maxls=100,
    )
    grad_inf = np.linalg.norm(info["grad"], ord=np.inf)
    if (info["warnflag"] != 0 or not np.isfinite(grad_inf)
            or grad_inf > PGRAD_TOL):
        raise RuntimeError("L-BFGS-B did not meet the gradient tolerance: "
                           f"task={info['task']}, grad_inf={grad_inf:.3e}")
    return a_min, E_min, info


def solution_diagnostics(u, info):
    """Return stationarity and monotonicity checks not imposed by build_u."""
    v = x + u
    el_residual = (alpha * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / dx**2 -
                   Wprime(v))
    dv_forward = 1.0 + (np.roll(u, -1) - u) / dx
    return {
        "reduced gradient inf norm": np.linalg.norm(info["grad"], ord=np.inf),
        "Euler-Lagrange residual inf norm": np.linalg.norm(el_residual,
                                                           ord=np.inf),
        "minimum forward derivative of v": np.min(dv_forward),
    }


def get_phase_fixed_solution():
    """Run deterministic multistart and return the lowest-energy solution.

    Agreement between starts supports robustness of the computed discrete
    minimizer but does not prove global minimality of the nonconvex objective.
    """
    runs = []
    for name, a0 in initial_guesses().items():
        a_min, E_min, info = solve_from_initial_guess(a0)
        runs.append((name, a_min, E_min, info))

    best_name, a_min, E_min, info = min(runs, key=lambda run: run[2])
    u = build_u(a_min)
    v = x + u
    diagnostics = solution_diagnostics(u, info)

    residual = diagnostics["Euler-Lagrange residual inf norm"]
    min_derivative = diagnostics["minimum forward derivative of v"]
    if residual > EL_RESIDUAL_TOL or min_derivative <= 0.0:
        raise RuntimeError(
            "Computed profile failed the solution diagnostics: "
            f"residual={residual:.3e}, min_dv={min_derivative:.3e}")

    return u, v, E_min, info, best_name, runs, diagnostics


if __name__ == "__main__":
    u, v, E_min, info, best_name, runs, diagnostics = (
        get_phase_fixed_solution())

    outdir = (Path(__file__).resolve().parents[2] / "outputs" / "relaxation" /
              "output_jv_phase_fixed_finite_difference_lbfgs_single_case")
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / "u_phase_fixed.npy", u)
    np.save(outdir / "v_phase_fixed.npy", v)

    energies = [run[2] for run in runs]
    print("Multistart results (evidence, not a proof of global minimality):")
    for name, _, energy, run_info in runs:
        grad_inf = np.linalg.norm(run_info["grad"], ord=np.inf)
        print(f"  {name:>13}: E={energy:.15e}, grad_inf={grad_inf:.3e}")
    print(f"Energy spread: {max(energies) - min(energies):.3e}")
    print("Selected start:", best_name)
    print("Optimizer task:", info["task"])
    print(f"Parameters: N={N}, alpha={alpha}, W(s)=2*cos(s)")
    print("Transformed discrete energy E[u]:", E_min)
    print("Discrete Equation (6) energy J[v]:", E_min + alpha * np.pi)
    print("Solution diagnostics:")
    for name, value in diagnostics.items():
        print(f"  {name}: {value:.3e}")

    x_plot = np.append(x, np.pi)
    u_plot = np.append(u, u[0])
    v_plot = np.append(v, v[0] + 2.0 * np.pi)

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, u_plot, linewidth=2)
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.title("Odd periodic correction $u$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / "u_phase_fixed.png", dpi=150)

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, v_plot, linewidth=2)
    plt.xlabel("$x$")
    plt.ylabel("$v(x)$")
    plt.title("Phase-fixed minimizer $v=x+u$ with $v(0)=0$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / "v_phase_fixed.png", dpi=150)

    plt.show()
