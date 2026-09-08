"""Compute one finite-difference continuum-energy minimizer.

Method:
    Centered periodic finite differences and L-BFGS-B optimization.
Purpose:
    Minimize the 200-point continuum energy for one epsilon and plot the result.
Inputs:
    In-file values ``N=200`` and ``eps=0.1`` with a zero initial displacement.
Outputs:
    ``u_min.npy``, a minimizer PDF, and a modified-disregistry PDF.
Output location:
    ``outputs/relaxation/output_finite_difference_lbfgs_single_case``.
Dependencies:
    NumPy, SciPy, and Matplotlib; no local module imports.
Related files:
    Its 200-point ``u_min.npy`` is the input to the finite-difference band study.
"""

# Originally written by David Zhou during his University of Minnesota REU in
# summer 2024 and revised by ABW in spring 2025.

### IMPORTS ###

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

### VARIABLES ###

# number of grid points over interval 0 to 2pi
N = 200
# epsilon
eps = .1

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "outputs"
    / "relaxation"
    / "output_finite_difference_lbfgs_single_case"
)

### FUNCTIONS ###


def I(x, dx, u, N, eps):
    # this function computes the total energy
    # x is vector of gridpoints, dx is grid spacing, u is vector of u values at gridpoints, N is number of gridpoints, eps is parameter epsilon
    # compute total energy by summing over gridpoints
    I = 0
    for i in range(N):
        # elasticity part
        I += ((eps**2) / 2) * (((u[(i + 1) % N] - u[(i - 1) % N]) /
                                (2 * dx))**2) * dx
        # stacking part, note minus sign
        I += -np.cos(x[i] + u[i]) * dx
    # return total energy
    return I


def grad_I(x, dx, u, N, eps):
    # this function computes the gradient of the energy with respect to u
    # x is vector of gridpoints, dx is grid spacing, u is vector of u values at gridpoints, N is number of gridpoints, eps is parameter epsilon
    # compute gradient vector
    grad_I = np.zeros(N)
    for i in range(N):
        # elasticity part
        grad_I[i] += (eps**2) * ((2 * u[i] - u[(i - 2) % N] - u[(i + 2) % N]) /
                                 (4 * dx))
        # stacking part
        grad_I[i] += np.sin(x[i] + u[i]) * dx
    # return gradient vector
    return grad_I


def get_u(x, dx, N, eps):
    # this function computes minimizers of the energy using a kind of gradient descent
    # x is vector of gridpoints, dx is grid spacing, N is number of gridpoints, eps is parameter epsilon
    # make I into a callable function of u
    I_test = lambda u: I(x, dx, u, N, eps)
    # make I_grad into a callable function of u
    grad_I_test = lambda u: grad_I(x, dx, u, N, eps)
    # initial guess
    u0 = np.zeros(N)
    # compute minimizer
    u_min = opt.fmin_l_bfgs_b(I_test, u0, grad_I_test)
    # return minimizer
    return u_min[0]


### MAIN ###

# precompute grid spacing
dx = 2 * np.pi / N
# precompute vector of gridpoints
x = np.linspace(0, 2 * np.pi, N, endpoint=False)
# compute minimizer
u = get_u(x, dx, N, eps)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# save minimizer as pickle file
with (OUTPUT_DIR / 'u_min.npy').open('wb') as f:
    np.save(f, u)
# plot minimizer
plt.figure()
plt.plot(x, u)
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.title('Minimizer $u$')
plt.savefig(OUTPUT_DIR / (f'minimizer_eps_%s.pdf' % eps))
# plot modified disregistry function
plt.figure()
plt.plot(x, x + u)
plt.xlabel('$x$')
plt.ylabel('$x+u$')
plt.title('Modified disregistry $x+u$')
plt.savefig(OUTPUT_DIR / (f'modified_disregistry_eps_%s.pdf' % eps))
# show plots
plt.show()
