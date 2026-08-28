import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# number of grid points
N = 400

# choose this depending on which model you want:
# 1) exact paper functional J[v] = ∫ (1/2)|v'|^2 + cos(v) dx  --> alpha = 1.0
# 2) epsilon-family analogue                        --> alpha = eps**2
eps = 0.5
alpha = eps**2  # use alpha = eps**2 if you want an epsilon-scaled family

# grid on [-pi, pi)
x = np.linspace(-np.pi, np.pi, N, endpoint=False)
dx = 2 * np.pi / N


def energy_u(u):
    """
    Minimize in the periodic variable u, where v = x + u.
    Then J[v] is equivalent (up to an additive constant) to

        E[u] = ∫ (alpha/2) |u'|^2 + cos(x+u) dx

    when W(v)=cos(v).
    """
    du = np.roll(u, -1) - u  # forward difference, periodic
    elasticity = 0.5 * alpha * (du / dx)**2
    stacking = np.cos(x + u)
    return np.sum(elasticity + stacking) * dx


def grad_energy_u(u):
    """
    Gradient of the discrete energy above.
    """
    grad_elasticity = alpha * (2 * u - np.roll(u, 1) - np.roll(u, -1)) / dx
    grad_stacking = -np.sin(x + u) * dx
    return grad_elasticity + grad_stacking


def get_u():
    # symmetric initial guess helps select the centered odd minimizer
    u0 = np.zeros(N)

    u_min, E_min, info = opt.fmin_l_bfgs_b(energy_u,
                                           u0,
                                           fprime=grad_energy_u,
                                           pgtol=1e-10,
                                           maxiter=500)

    print("Optimizer info:", info["task"])
    print(f"Parameters: eps={eps}, alpha={alpha}")
    print("Final energy:", E_min)
    return u_min


if __name__ == "__main__":
    u = get_u()
    v = x + u

    np.save("u_min.npy", u)
    np.save("v_min.npy", v)

    plt.figure(figsize=(8, 5))
    plt.plot(x, v, linewidth=2, label="Minimizer $v=x+u$")
    plt.xlabel("$x$")
    plt.ylabel("$v(x)$")
    plt.title("Energy Minimizer $v$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("minimizer_v.pdf")

    plt.figure(figsize=(8, 5))
    plt.plot(x, u, linewidth=2, label="Periodic correction $u$")
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.title("Periodic Variable $u$")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("minimizer_u.pdf")

    plt.show()
