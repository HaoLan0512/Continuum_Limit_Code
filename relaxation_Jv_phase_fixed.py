from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# ============================================================
# Numerically minimize
#     J[v] = ∫_{-π}^{π} [ 1/2 |v'(x)|^2 + W(v(x)) ] dx,
# over v ∈ x + H^1_per.
#
# We write v(x) = x + u(x), where u is 2π-periodic, so (up to
# an additive constant) this is equivalent to minimizing
#     E[u] = ∫_{-π}^{π} [ 1/2 |u'(x)|^2 + W(x+u(x)) ] dx.
#
# To match the phase-fixed minimizer from Theorem 1, we restrict
# to odd periodic u. Then v = x + u is automatically odd and
# satisfies v(0) = 0.
# ============================================================

N = 400  # must be even
assert N % 2 == 0

# Choose this depending on which model you want:
# 1) exact paper functional J[v] = ∫ (1/2)|v'|^2 + cos(v) dx  --> alpha = 1.0
# 2) epsilon-family analogue                                  --> alpha = eps**2
eps = 0.5
alpha = eps**2  # set alpha = 1.0 to recover the exact paper functional

mid = N // 2
x = np.linspace(-np.pi, np.pi, N, endpoint=False)
dx = 2 * np.pi / N


def W(s):
    return np.cos(s)


def Wprime(s):
    return -np.sin(s)


def build_u(a):
    """
    Build an odd periodic grid function u from the free variables a.

    Grid convention:
        x[0]   = -π,
        x[mid] = 0,
        x[N-i] = -x[i].

    For odd periodic u we must have
        u[0] = 0,
        u[mid] = 0,
        u[N-i] = -u[i],   i = 1,...,mid-1.
    """
    u = np.zeros(N)
    u[1:mid] = a
    u[mid + 1:] = -a[::-1]
    return u


def reduced_gradient(full_grad):
    """
    Chain rule for the odd-periodic parametrization.

    If a_k = u[k+1] and u[N-k-1] = -a_k, then
        dE/da_k = dE/du[k+1] - dE/du[N-k-1].
    """
    return full_grad[1:mid] - full_grad[:mid:-1]


def energy_full(u):
    """
    Discrete energy for the periodic variable u.
    Uses forward differences to avoid the even/odd decoupling created
    by the centered-difference square.
    """
    du = np.roll(u, -1) - u
    elasticity = 0.5 * alpha * (du / dx)**2
    stacking = W(x + u)
    return np.sum(elasticity + stacking) * dx


def grad_full(u):
    """
    Gradient of energy_full with respect to the full vector u.
    """
    grad_elasticity = alpha * (2 * u - np.roll(u, 1) - np.roll(u, -1)) / dx
    grad_stacking = Wprime(x + u) * dx
    return grad_elasticity + grad_stacking


def energy_reduced(a):
    u = build_u(a)
    return energy_full(u)


def grad_reduced(a):
    u = build_u(a)
    g = grad_full(u)
    return reduced_gradient(g)


def get_phase_fixed_solution():
    # Start from the odd zero guess; the parametrization keeps all iterates odd.
    a0 = np.zeros(mid - 1)

    a_min, E_min, info = opt.fmin_l_bfgs_b(
        energy_reduced,
        a0,
        fprime=grad_reduced,
        pgtol=1e-12,
        maxiter=1000,
    )

    u = build_u(a_min)
    v = x + u
    return u, v, E_min, info


def constraint_report(u, v):
    odd_err_u = np.max(np.abs(u[1:mid] + u[:mid:-1]))
    seam_err_u = max(abs(u[0]), abs(u[mid]))
    odd_err_v = np.max(np.abs(v[1:mid] + v[:mid:-1]))
    seam_err_v = max(abs(v[0] + np.pi), abs(v[mid]))
    return {
        "oddness error in u": odd_err_u,
        "seam/phase error in u": seam_err_u,
        "oddness error in v": odd_err_v,
        "seam/phase error in v": seam_err_v,
    }


if __name__ == "__main__":
    u, v, E_min, info = get_phase_fixed_solution()

    outdir = Path(__file__).resolve().parent / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / "u_phase_fixed.npy", u)
    np.save(outdir / "v_phase_fixed.npy", v)

    print("Optimizer task:", info.get("task"))
    print(f"Parameters: eps={eps}, alpha={alpha}")
    print("Final energy:", E_min)
    print("Constraint report:")
    for k, val in constraint_report(u, v).items():
        print(f"  {k}: {val:.3e}")

    plt.figure(figsize=(8, 5))
    plt.plot(x, u, linewidth=2)
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.title("Odd periodic correction $u$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / "u_phase_fixed.png", dpi=150)

    plt.figure(figsize=(8, 5))
    plt.plot(x, v, linewidth=2)
    plt.xlabel("$x$")
    plt.ylabel("$v(x)$")
    plt.title("Phase-fixed minimizer $v=x+u$ with $v(0)=0$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / "v_phase_fixed.png", dpi=150)

    plt.show()
