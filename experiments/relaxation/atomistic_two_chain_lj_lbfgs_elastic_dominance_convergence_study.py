"""Compare the paper's atomistic and continuum two-chain minimizers.

Method:
    Equations (12)-(13), a symmetric Lennard-Jones image cutoff, exact
    finite-cutoff gradients, phase-fixed continuum finite differences, and
    deterministic L-BFGS-B solves from a continuum warm start and one fixed
    smooth random-Fourier phase-stress start, with one conditional Newton
    correction and analytic curvature screens.
Purpose:
    Test the predicted O(N^-1) atomistic-to-continuum convergence rate in the
    layer-weighted discrete H^2 and maximum norms.
Inputs:
    N=20, 40, 80, 160, 320, 640; matched effective LJ parameters; M=80 images.
Outputs:
    One CSV diagnostic table, one NPZ profile archive, one PNG convergence
    figure, and a compact check report.
Output location:
    outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_elastic_dominance_convergence_study.
Dependencies:
    NumPy, SciPy, and Matplotlib; clr.potentials.lj_periodic for the effective
    LJ pair potential; clr.relaxation.odd_periodic for the continuum reduction.
Related files:
    jv_phase_fixed_finite_difference_lbfgs_single_case.py uses the same shared
    continuum odd-coordinate construction.
"""

import sys
import csv
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from scipy.interpolate import CubicSpline

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from clr.potentials.lj_periodic import (
    EffectiveLJParameters,
    pair_potential,
    pair_potential_prime,
    pair_potential_second,
    periodized_W,
    periodized_Wprime,
)
from clr.relaxation.odd_periodic import (
    build_odd_periodic as build_odd_q,
    reduce_odd_gradient,
)

PI = np.pi
TWOPI = 2.0 * PI

N_VALUES = (20, 40, 80, 160, 320, 640)
PAIR_CUTOFF = 80
CUTOFF_CHECK = 160
CONTINUUM_POINTS = (400, 800)

LATTICE_CONSTANT = 1.0
LJ_SIGMA = 0.9
INTERLAYER_DISTANCE = 1.0
LJ_EPSILON = 0.5
LJ_PARAMETERS = EffectiveLJParameters(
    epsilon=LJ_EPSILON,
    sigma=LJ_SIGMA,
    interlayer_distance=INTERLAYER_DISTANCE,
    lattice_constant=LATTICE_CONSTANT,
)

ATOMISTIC_MEMORY = 50
ATOMISTIC_PGTOL = 5.0e-8
ATOMISTIC_EL_OVER_H_TOL = 1.0e-2
ATOMISTIC_RANDOM_FOURIER_SEED = 3
ATOMISTIC_RANDOM_FOURIER_MODES = 5
ATOMISTIC_RANDOM_MAX_AMPLITUDE = 0.1
ATOMISTIC_RANDOM_START_NAME = f"random_fourier_seed_{ATOMISTIC_RANDOM_FOURIER_SEED}"
ATOMISTIC_RANDOM_SEED_TAG = f"seed{ATOMISTIC_RANDOM_FOURIER_SEED}"
ATOMISTIC_START_NAMES = ("sampled_continuum", ATOMISTIC_RANDOM_START_NAME)
CONTINUUM_GRAD_TOL = 2.0e-7
CONTINUUM_EL_TOL = 2.0e-5
MAX_CONTINUUM_SOLVER_CALLS = 3
MAX_ITERATIONS = 5000
MAX_LINESEARCH = 100

# Practical soft-curvature band for this objective, not a proved stability gap.
CURVATURE_SOFT_TOL = 1.0e-8
NEWTON_MAX_STEP_OVER_H = 0.1
NEWTON_ENERGY_ROUNDOFF_FACTOR = 64.0
ENERGY_COMPARISON_RTOL = 1.0e-10

GRADIENT_CHECK_STEP = 1.0e-6
GRADIENT_CHECK_TOL = 1.0e-8
MEAN_TOL = 1.0e-12
REFINEMENT_FRACTION = 0.05  #check the (400,800) on the 640 atomistic grid, less than 5% means the error cause by continuum sampling is negligible
CUTOFF_EL_OVER_H_TOL = 1.0e-4
SLOPE_INTERVAL = (-1.2, -0.8)
REFLECTION_TIE_TOL = 1.0e-12


def V(s):
    """Return the even effective LJ pair potential in paper coordinates."""
    return pair_potential(s, LJ_PARAMETERS)


def Vprime(s):
    """Return the derivative of V with respect to its paper coordinate."""
    return pair_potential_prime(s, LJ_PARAMETERS)


def Vsecond(s):
    """Return the second derivative of V in paper coordinates."""
    return pair_potential_second(s, LJ_PARAMETERS)


def W(s, cutoff=PAIR_CUTOFF):
    """Return W(s)=4 sum_m V(s-2*pi*m) with a symmetric cutoff."""
    return periodized_W(s, LJ_PARAMETERS, cutoff)


def Wprime(s, cutoff=PAIR_CUTOFF):
    """Return the derivative of the truncated matched continuum potential."""
    return periodized_Wprime(s, LJ_PARAMETERS, cutoff)


def continuum_grid(point_count):
    """Return the even periodic grid x_j=-pi+j*dx and its spacing."""
    if point_count % 2:
        raise ValueError("The phase-fixed continuum grid must be even.")
    return np.linspace(-PI, PI, point_count,
                       endpoint=False), TWOPI / point_count


def continuum_full_value_gradient(q,
                                  x,
                                  dx,
                                  potential=W,
                                  potential_prime=Wprime):
    """Return the matched continuum correction energy and full gradient."""
    difference = np.roll(q, -1) - q
    energy = dx * np.sum(0.5 * (difference / dx)**2 + potential(x + q))
    gradient = ((2.0 * q - np.roll(q, 1) - np.roll(q, -1)) / dx +
                dx * potential_prime(x + q))
    return float(energy), gradient


def periodic_elastic_hessian(point_count, spacing):
    """Return the Hessian of the periodic forward-bond elastic energy."""
    indices = np.arange(point_count)
    hessian = np.eye(point_count) * (2.0 / spacing)
    hessian[indices, (indices + 1) % point_count] -= 1.0 / spacing
    hessian[indices, (indices - 1) % point_count] -= 1.0 / spacing
    return hessian


def continuum_full_hessian(q, x, dx, cutoff=PAIR_CUTOFF):
    """Return the matched LJ Hessian on all periodic continuum grid variables."""
    arguments = ((x + q)[:, None] - TWOPI *
                 np.arange(-cutoff, cutoff + 1)[None, :])
    wsecond = 4.0 * np.sum(Vsecond(arguments), axis=1)
    return periodic_elastic_hessian(q.size, dx) + np.diag(dx * wsecond)


def hessian_spectrum(hessian):
    """Return eigenpairs and a numerical curvature classification at this point."""
    unavailable = {
        "curvature_min_eigenvalue": np.nan,
        "curvature_tolerance": np.nan,
        "curvature_status": "unavailable",
    }
    if not np.all(np.isfinite(hessian)):
        return np.empty(0), np.empty((hessian.shape[0], 0)), unavailable
    try:
        eigenvalues, eigenvectors = la.eigh(hessian)
    except la.LinAlgError:
        return np.empty(0), np.empty((hessian.shape[0], 0)), unavailable
    minimum = float(eigenvalues[0])
    vector = eigenvectors[:, 0]
    residual = np.linalg.norm(hessian @ vector - minimum * vector)
    tolerance = max(
        CURVATURE_SOFT_TOL,
        100.0 * np.finfo(float).eps * max(1.0, np.linalg.norm(hessian, np.inf))
        + 10.0 * residual)
    status = ("negative" if minimum < -tolerance else
              "positive" if minimum > tolerance else "soft")
    return eigenvalues, eigenvectors, {
        "curvature_min_eigenvalue": minimum,
        "curvature_tolerance": float(tolerance),
        "curvature_status": status,
    }


def continuum_reduced_value_gradient(a,
                                     x,
                                     dx,
                                     potential=W,
                                     potential_prime=Wprime):
    """Return F(a)=E(build_odd_q(a)) and its reduced gradient."""
    q = build_odd_q(a, x.size)
    energy, full_gradient = continuum_full_value_gradient(
        q, x, dx, potential, potential_prime)
    return energy, reduce_odd_gradient(full_gradient)


def continuum_diagnostics(q, x, dx, potential_prime=Wprime):
    """Return monotonicity and Euler-Lagrange diagnostics."""
    residual = ((np.roll(q, -1) - 2.0 * q + np.roll(q, 1)) / dx**2 -
                potential_prime(x + q))
    return {
        "el_residual_inf_norm": float(np.max(np.abs(residual))),
        "min_forward_derivative":
        float(np.min(1.0 + (np.roll(q, -1) - q) / dx)),
    }


def solve_continuum_initial(initial,
                            x,
                            dx,
                            potential=W,
                            potential_prime=Wprime,
                            gradient_tol=CONTINUUM_GRAD_TOL,
                            el_residual_tol=CONTINUUM_EL_TOL):
    """Solve the continuum zero start, continuing at most twice after FACTR.
    The following checks were performed:
    1. The L-BFGS-B solver must not return a warning flag.
    2. The returned energy, point (inf norm), and gradient must be finite.
    3. The returned gradient inf norm must be below the specified tolerance.
    4. The returned Euler-Lagrange residual inf norm must be below the specified tolerance.
    5. The returned minimum forward derivative must be positive.
    """
    point = np.array(initial, dtype=float, copy=True)
    total_iterations = 0
    total_function_calls = 0
    for solver_call in range(1, MAX_CONTINUUM_SOLVER_CALLS + 1):
        point, energy, info = opt.fmin_l_bfgs_b(
            continuum_reduced_value_gradient,
            point,
            args=(x, dx, potential, potential_prime),
            pgtol=gradient_tol,
            factr=1.0,
            maxiter=MAX_ITERATIONS,
            maxls=MAX_LINESEARCH,
        )
        total_iterations += int(info["nit"])
        total_function_calls += int(info["funcalls"])
        gradient_inf = float(np.max(np.abs(info["grad"])))
        q = build_odd_q(point, x.size)
        diagnostics = continuum_diagnostics(q, x, dx, potential_prime)
        task = str(info["task"])
        if (info["warnflag"] != 0 or not np.isfinite(energy)
                or not np.all(np.isfinite(point))
                or not np.isfinite(gradient_inf)):
            raise RuntimeError(
                f"Continuum L-BFGS-B failed: task={task}, grad={gradient_inf:.3e}"
            )
        accepted = (gradient_inf <= gradient_tol
                    and diagnostics["el_residual_inf_norm"] <= el_residual_tol
                    and diagnostics["min_forward_derivative"] > 0.0)
        if accepted:
            return {
                "q": q,
                "energy": float(energy),
                "gradient_inf_norm": gradient_inf,
                "iterations": total_iterations,
                "function_calls": total_function_calls,
                "solver_calls": solver_call,
                "task": task,
                **diagnostics,
            }
        if "RELATIVE REDUCTION OF F" not in task:
            raise RuntimeError(
                "Continuum solve missed its acceptance tolerances: "
                f"task={task}, grad={gradient_inf:.3e}, "
                f"residual={diagnostics['el_residual_inf_norm']:.3e}")
    raise RuntimeError(
        "Continuum FACTR restarts missed the acceptance tolerances: "
        f"grad={gradient_inf:.3e}, residual={diagnostics['el_residual_inf_norm']:.3e}"
    )


def solve_continuum_reference(point_count,
                              potential=W,
                              potential_prime=Wprime,
                              gradient_tol=CONTINUUM_GRAD_TOL,
                              el_residual_tol=CONTINUUM_EL_TOL):
    """Return one deterministic phase-fixed continuum reference."""
    x, dx = continuum_grid(point_count)
    initial = np.zeros(point_count // 2 - 1)
    solution = solve_continuum_initial(initial, x, dx, potential,
                                       potential_prime, gradient_tol,
                                       el_residual_tol)
    return {"point_count": point_count, "x": x, "dx": dx, **solution}


def periodic_q_spline(x, q):
    """Return the periodic cubic interpolant of the correction q only."""
    return CubicSpline(np.append(x, PI),
                       np.append(q, q[0]),
                       bc_type="periodic")


def wrap_to_paper_cell(x):
    """Wrap coordinates to the half-open paper cell [-pi,pi)."""
    return (np.asarray(x) + PI) % TWOPI - PI


# ------------ Atomistic model begins here ---------------------
def atomistic_geometry(N):
    """Return N1, N2, h1, h2, and the paper's harmonic spacing h."""
    N1, N2 = N, N + 1
    h1, h2 = TWOPI / N1, TWOPI / N2
    h = 2.0 * h1 * h2 / (h1 + h2)
    return N1, N2, h1, h2, h


def split_layers(u, N):
    """Split a length-(2N+1) vector into its N and N+1 layers."""
    return u[:N], u[N:]


def project_mean_gauge(u, N):
    """Project a full state to mean(u1)+mean(u2)=0 usiing symmetry c, with d=0 in the paper."""
    _, _, h1, h2, _ = atomistic_geometry(N)
    u1, u2 = split_layers(np.asarray(u, dtype=float), N)
    c = (np.mean(u1) + np.mean(u2)) / (h1 + h2)
    return np.concatenate((u1 - h2 * c, u2 - h1 * c))


def build_mean_gauge(a, N):
    """Reconstruct a mean-gauged full state from exactly 2N variables by using the mean 0 condition to determine the last variable of the second layer."""
    a = np.asarray(a, dtype=float)
    if a.size != 2 * N:
        raise ValueError(
            f"Expected {2 * N} atomistic variables, got {a.size}.")
    u1 = a[:N]
    u2_free = a[N:]
    u2_last = -(N + 1) * np.mean(u1) - np.sum(u2_free)
    return np.concatenate((u1, u2_free, np.array([u2_last])))


def reduce_mean_gauge_gradient(full_gradient, N):
    """Apply the transpose chain rule of the 2N-variable gauge map."""
    g1, g2 = split_layers(full_gradient, N)
    return np.concatenate((
        g1 - ((N + 1) / N) * g2[-1],
        g2[:-1] - g2[-1],
    ))


def atomistic_reduced_coordinates(u, N):
    """Convert a full state to the 2N coordinates used by L-BFGS-B.

    ``build_mean_gauge(atomistic_reduced_coordinates(u, N), N) ==
    project_mean_gauge(u, N)``.
    """
    gauged = project_mean_gauge(u, N)
    u1, u2 = split_layers(gauged, N)
    return np.concatenate((u1, u2[:-1]))


def raw_atomistic_value_gradient(u, N, cutoff):
    """Return truncated Equations (12)-(13) and their exact full gradient."""
    N1, N2, h1, h2, h = atomistic_geometry(N)
    u1, u2 = split_layers(np.asarray(u, dtype=float), N)
    difference1 = np.roll(u1, -1) - u1
    difference2 = np.roll(u2, -1) - u2
    energy = (0.5 * np.sum(difference1**2) / h1 +
              0.5 * np.sum(difference2**2) / h2)
    g1 = (2.0 * u1 - np.roll(u1, 1) - np.roll(u1, -1)) / h1
    g2 = (2.0 * u2 - np.roll(u2, 1) - np.roll(u2, -1)) / h2
    image_indices = np.arange(-cutoff, cutoff + 1)

    layer_data = (
        (u1, u2, N1, N2, h1, h2, 1.0, g1, g2),
        (u2, u1, N2, N1, h2, h1, -1.0, g2, g1),
    )
    for ui, opposite, Ni, No, hi, ho, sign, gi, go in layer_data:
        n = np.arange(Ni)[:, None]
        m = image_indices[None, :]
        opposite_indices = (n + m) % No
        arguments = (sign * h * n + (h / ho) * ui[:, None] -
                     (h / hi) * opposite[opposite_indices] - TWOPI *
                     (h / hi) * m)
        pair_values = V(arguments)
        pair_derivatives = Vprime(arguments)
        energy += hi * np.sum(pair_values)
        gi += hi * (h / ho) * np.sum(pair_derivatives, axis=1)
        np.add.at(go, opposite_indices.ravel(),
                  (-h * pair_derivatives).ravel())
    return float(energy), np.concatenate((g1, g2))


def atomistic_value_gradient(a, N, cutoff):
    """Evaluate the mean-gauged objective and its exact chain-rule gradient."""
    u = build_mean_gauge(a, N)
    energy, full_gradient = raw_atomistic_value_gradient(u, N, cutoff)
    return energy, reduce_mean_gauge_gradient(full_gradient, N)


def raw_atomistic_hessian(u, N, cutoff):
    """Differentiate the same two directed image sums as the full gradient."""
    N1, N2, h1, h2, h = atomistic_geometry(N)
    u1, u2 = split_layers(u, N)
    hessian = la.block_diag(periodic_elastic_hessian(N1, h1),
                           periodic_elastic_hessian(N2, h2))
    images = np.arange(-cutoff, cutoff + 1)[None, :]
    layer_data = (
        (u1, u2, N1, N2, h1, h2, 1.0, 0, N1),
        (u2, u1, N2, N1, h2, h1, -1.0, N1, 0),
    )
    for ui, opposite, Ni, No, hi, ho, sign, offset, other_offset in layer_data:
        n = np.arange(Ni)[:, None]
        opposite_indices = (n + images) % No
        alpha, beta = h / ho, h / hi
        arguments = (sign * h * n + alpha * ui[:, None] -
                     beta * opposite[opposite_indices] - TWOPI * beta * images)
        weights = hi * Vsecond(arguments)
        row = np.broadcast_to(n + offset, weights.shape).ravel()
        column = (opposite_indices + other_offset).ravel()
        weights = weights.ravel()
        # Each image contributes hi*V''(A)*(alpha*e_i-beta*e_j)^(outer 2).
        np.add.at(hessian, (row, row), alpha**2 * weights)
        np.add.at(hessian, (column, column), beta**2 * weights)
        np.add.at(hessian, (row, column), -alpha * beta * weights)
        np.add.at(hessian, (column, row), -alpha * beta * weights)
    return hessian


def mean_gauge_reflector(N):
    """Return v with Q=(I-2vv.T)[:,:-1] orthonormal in the mean-gauge tangent."""
    normal = np.r_[np.full(N, 1.0 / N), np.full(N + 1, 1.0 / (N + 1))]
    normal /= np.linalg.norm(normal)
    normal[-1] += 1.0
    return normal / np.linalg.norm(normal)


def atomistic_tangent_hessian(u, N, cutoff):
    """Return Q.T H Q (2N by 2N) and its Householder reflector, without forming Q."""
    hessian = raw_atomistic_hessian(u, N, cutoff)
    reflector = mean_gauge_reflector(N)
    product = hessian @ reflector
    transformed = (hessian - 2.0 * np.outer(reflector, product)
                   - 2.0 * np.outer(product, reflector)
                   + 4.0 * np.dot(reflector, product)
                   * np.outer(reflector, reflector))
    return transformed[:-1, :-1], reflector


def sample_continuum_pair(spline, N, phase_shift=0.0):
    """Sample (q(x-delta)-delta)/2 and its negative on the native grids.

    The coupled shift is the continuum translation symmetry, with q=v_0-x.
    It changes the reference only; it is not an atomistic transformation.
    """
    N1, N2, h1, h2, _ = atomistic_geometry(N)
    x1 = h1 * np.arange(N1)
    x2 = h2 * np.arange(N2)
    q1 = spline(wrap_to_paper_cell(x1 - phase_shift)) - phase_shift
    q2 = spline(wrap_to_paper_cell(x2 - phase_shift)) - phase_shift
    return np.concatenate((0.5 * q1, -0.5 * q2))


def continuum_phase_mode_overlap(u, N, spline, cutoff):
    """Return (second eigenvalue, phase overlap) for a raw 2N+1 endpoint."""
    if not np.all(np.isfinite(u)):
        return np.nan, np.nan
    hessian, reflector = atomistic_tangent_hessian(u, N, cutoff)
    if not np.all(np.isfinite(hessian)):
        return np.nan, np.nan
    try:
        values, vectors = la.eigh(hessian, subset_by_index=(0, 1))
    except la.LinAlgError:
        return np.nan, np.nan

    N1, N2, h1, h2, _ = atomistic_geometry(N)
    u1, u2 = split_layers(u, N)
    delta = np.mean(u2) - np.mean(u1)
    derivative = spline.derivative()
    # Differentiate q(x-delta)-delta, then split between the two layers.
    phase = np.concatenate((
        -0.5 * (derivative(wrap_to_paper_cell(h1 * np.arange(N1) - delta)) + 1.0),
        0.5 * (derivative(wrap_to_paper_cell(h2 * np.arange(N2) - delta)) + 1.0),
    ))
    phase = project_mean_gauge(phase, N)
    phase = (phase - 2.0 * reflector * np.dot(reflector, phase))[:-1]
    phase_norm = np.linalg.norm(phase)
    if not np.isfinite(phase_norm) or phase_norm == 0.0:
        return float(values[1]), np.nan
    overlap = abs(np.dot(vectors[:, 0], phase)) / phase_norm
    return float(values[1]), float(np.clip(overlap, 0.0, 1.0))


def random_fourier_atomistic_start(N):
    """Return the configured mean-gauged smooth random-Fourier start.

    The same continuous Fourier coefficients are sampled on every
    atomistic refinement. Opposite constant modes probe relative registry,
    while independent nonzero modes leave both layer fields unrestricted.
    """
    N1, N2, h1, h2, _ = atomistic_geometry(N)
    x1 = h1 * np.arange(N1)
    x2 = h2 * np.arange(N2)
    modes = np.arange(1, ATOMISTIC_RANDOM_FOURIER_MODES + 1, dtype=float)
    rng = np.random.default_rng(ATOMISTIC_RANDOM_FOURIER_SEED)
    relative_offset = rng.normal()
    sine_coefficients = rng.normal(size=(2, modes.size)) / modes**2
    cosine_coefficients = rng.normal(size=(2, modes.size)) / modes**2

    u1 = np.full(N1, relative_offset)
    u2 = np.full(N2, -relative_offset)
    for index, mode in enumerate(modes):
        u1 += (sine_coefficients[0, index] * np.sin(mode * x1) +
               cosine_coefficients[0, index] * np.cos(mode * x1))
        u2 += (sine_coefficients[1, index] * np.sin(mode * x2) +
               cosine_coefficients[1, index] * np.cos(mode * x2))

    full = project_mean_gauge(np.concatenate((u1, u2)), N)
    full *= ATOMISTIC_RANDOM_MAX_AMPLITUDE / np.max(np.abs(full))
    return atomistic_reduced_coordinates(full, N)


def atomistic_initial_guesses(N, continuum_pair):
    """Return the independent warm and fixed phase-stress starts."""
    return {
        "sampled_continuum": atomistic_reduced_coordinates(continuum_pair, N),
        ATOMISTIC_RANDOM_START_NAME: random_fourier_atomistic_start(N),
    }


def atomistic_diagnostics(u, N, cutoff):
    """Evaluate energy, stationarity, and means on a full atomistic state."""
    energy, gradient = raw_atomistic_value_gradient(u, N, cutoff)
    _, _, h1, h2, h = atomistic_geometry(N)
    u1, u2 = split_layers(u, N)
    g1, g2 = split_layers(gradient, N)
    el_residual = float(max(np.max(np.abs(g1 / h1)), np.max(np.abs(g2 / h2))))
    return {
        "energy":
        float(energy),
        "reduced_gradient_inf_norm":
        float(np.max(np.abs(reduce_mean_gauge_gradient(gradient, N)))),
        "el_residual_inf_norm":
        el_residual,
        "el_residual_over_h":
        el_residual / h,
        "mean1":
        float(np.mean(u1)),
        "mean2":
        float(np.mean(u2)),
        "mean_sum":
        float(np.mean(u1) + np.mean(u2)),
    }


def atomistic_stationary(diagnostics):
    """Require finite reduced gradient and scaled EL residual at their targets."""
    gradient = diagnostics["reduced_gradient_inf_norm"]
    residual = diagnostics["el_residual_over_h"]
    return bool(np.isfinite(gradient) and np.isfinite(residual)
                and gradient <= ATOMISTIC_PGTOL
                and residual <= ATOMISTIC_EL_OVER_H_TOL)


def polish_atomistic_endpoint(u, N, cutoff, allow_correction=True):
    """Assess the raw endpoint and attempt at most one positive-spectrum Newton step."""
    diagnostics = atomistic_diagnostics(u, N, cutoff)
    hessian, reflector = atomistic_tangent_hessian(u, N, cutoff)
    values, vectors, curvature = hessian_spectrum(hessian)
    stationary = atomistic_stationary(diagnostics)
    correction = {
        "newton_correction_status": "not_needed" if stationary else "not_attempted",
        "newton_correction_applied": False,
        "newton_step_inf_norm": 0.0,
    }
    if not allow_correction or stationary:
        return u, diagnostics, curvature, correction
    if curvature["curvature_status"] not in ("soft", "positive"):
        correction["newton_correction_status"] = "blocked_curvature"
        return u, diagnostics, curvature, correction

    _, gradient = raw_atomistic_value_gradient(u, N, cutoff)
    tangent_gradient = (gradient - 2.0 * reflector * np.dot(reflector, gradient))[:-1]
    positive = values > curvature["curvature_tolerance"]
    tangent_step = -vectors[:, positive] @ (
        (vectors[:, positive].T @ tangent_gradient) / values[positive])
    step = np.r_[tangent_step, 0.0]
    step -= 2.0 * reflector * np.dot(reflector, step)
    step_norm = float(np.max(np.abs(step)))
    correction["newton_step_inf_norm"] = step_norm
    if (not np.all(np.isfinite(step))
            or step_norm > NEWTON_MAX_STEP_OVER_H * atomistic_geometry(N)[-1]):
        correction["newton_correction_status"] = "rejected_step"
        return u, diagnostics, curvature, correction

    candidate = u + step
    candidate_diagnostics = atomistic_diagnostics(candidate, N, cutoff)
    normalized, _ = appendix_b_normalize(candidate, N)
    normalized_diagnostics = atomistic_diagnostics(normalized, N, cutoff)
    energy_tolerance = (NEWTON_ENERGY_ROUNDOFF_FACTOR * np.finfo(float).eps
                        * max(1.0, abs(diagnostics["energy"]),
                              abs(candidate_diagnostics["energy"])))
    if (not np.isfinite(candidate_diagnostics["energy"])
            or not atomistic_stationary(candidate_diagnostics)
            or not atomistic_stationary(normalized_diagnostics)
            or candidate_diagnostics["energy"] > diagnostics["energy"] + energy_tolerance):
        correction["newton_correction_status"] = "rejected_acceptance"
        return u, diagnostics, curvature, correction

    hessian, _ = atomistic_tangent_hessian(candidate, N, cutoff)
    _, _, curvature = hessian_spectrum(hessian)
    correction["newton_correction_status"] = "applied"
    correction["newton_correction_applied"] = True
    return candidate, candidate_diagnostics, curvature, correction


def solve_atomistic_initial(initial, N, cutoff):
    """Run one L-BFGS-B call, optionally polish once, and assess both representatives."""
    initial_energy, initial_gradient = atomistic_value_gradient(initial, N, cutoff)
    initial_gradient_norm = float(np.max(np.abs(initial_gradient)))
    # run L-BFGS-B on the 2N reduced coordinates, with sum of means = 0
    point, _, info = opt.fmin_l_bfgs_b(
        atomistic_value_gradient,
        np.array(initial, dtype=float, copy=True),
        args=(N, cutoff),
        m=ATOMISTIC_MEMORY,
        pgtol=ATOMISTIC_PGTOL,
        factr=1.0,
        maxiter=MAX_ITERATIONS,
        maxls=MAX_LINESEARCH,
    )
    endpoint = build_mean_gauge(point, N)  #recover the 2N+1 full state
    if np.all(np.isfinite(endpoint)):
        endpoint, raw_diagnostics, curvature, correction = polish_atomistic_endpoint(
            endpoint, N, cutoff, allow_correction=info["warnflag"] == 0)
        normalized, normalization = appendix_b_normalize(
            endpoint, N)  #apply appendix-B normalization
        diagnostics = atomistic_diagnostics(
            normalized, N, cutoff
        )  # compute the energy, reduced gradient, and Euler-Lagrange residual of the normalized endpoint
    else:
        normalized = endpoint
        diagnostics = {
            "energy": np.nan,
            "reduced_gradient_inf_norm": np.nan,
            "el_residual_inf_norm": np.nan,
            "el_residual_over_h": np.nan,
            "mean1": np.nan,
            "mean2": np.nan,
            "mean_sum": np.nan,
        }
        normalization = {
            "normalization_d": np.nan,
            "normalization_c": np.nan,
            "normalization_mean_bound": atomistic_geometry(N)[-1] / 4.0,
            "mean1": np.nan,
            "mean2": np.nan,
            "mean_sum": np.nan,
        }
        raw_diagnostics = diagnostics.copy()
        curvature = {"curvature_min_eigenvalue": np.nan,
                     "curvature_tolerance": np.nan, "curvature_status": "unavailable"}
        correction = {"newton_correction_status": "not_attempted",
                      "newton_correction_applied": False, "newton_step_inf_norm": 0.0}
    #checks finiteness of the states and diagnostics.
    finite = (np.all(np.isfinite(point)) and np.all(np.isfinite(normalized))
              and all(np.isfinite(value) for value in diagnostics.values()))
    # checks the sum-of-means constraint and Appendix-B mean bounds
    mean_ok = (abs(diagnostics["mean_sum"]) <= MEAN_TOL
               and max(abs(diagnostics["mean1"]), abs(diagnostics["mean2"]))
               <= normalization["normalization_mean_bound"] + 1.0e-14)
    # A soft eigenvalue supports no positive stability-gap or global-minimum claim.
    accepted = (info["warnflag"] == 0 and finite and mean_ok and
                atomistic_stationary(raw_diagnostics) and atomistic_stationary(diagnostics)
                and np.isfinite(raw_diagnostics["energy"])
                and curvature["curvature_status"] in ("soft", "positive"))
    return {
        "raw_u": endpoint,
        "initial_energy": float(initial_energy),
        "initial_reduced_gradient_inf_norm": initial_gradient_norm,
        "raw_energy": raw_diagnostics["energy"],
        "raw_reduced_gradient_inf_norm": raw_diagnostics["reduced_gradient_inf_norm"],
        "raw_el_residual_over_h": raw_diagnostics["el_residual_over_h"],
        "gradient_reduction_ratio": (
            raw_diagnostics["reduced_gradient_inf_norm"] / initial_gradient_norm
            if initial_gradient_norm != 0.0 else None),
        "lbfgsb_reduced_gradient_inf_norm": float(np.max(np.abs(info["grad"]))),
        "normalized_u": normalized,
        "accepted": bool(accepted),
        "iterations": int(info["nit"]),
        "function_calls": int(info["funcalls"]),
        "optimizer_task": str(info["task"]),
        "warnflag": int(info["warnflag"]),
        "normalization": normalization,
        **curvature,
        **correction,
        **diagnostics,
    }


def appendix_b_normalize(u, N):
    """Apply the paper's discrete integer-shift and common-translation gauge."""
    _, _, h1, h2, h = atomistic_geometry(N)
    u1, u2 = split_layers(u, N)
    mean1, mean2 = float(np.mean(u1)), float(np.mean(u2))
    d = int(np.rint((h2 * mean2 - h1 * mean1) / (h1 * h2)))
    c = -(mean1 + mean2 + h2 * d) / (h1 + h2)
    normalized = np.concatenate((
        np.roll(u1, -d) + h2 * d + h2 * c,
        np.roll(u2, -d) + h1 * c,
    ))
    normalized1, normalized2 = split_layers(normalized, N)
    return normalized, {
        "normalization_d": d,
        "normalization_c": float(c),
        "normalization_mean_bound": float(h / 4.0),
        "mean1": float(np.mean(normalized1)),
        "mean2": float(np.mean(normalized2)),
        "mean_sum": float(np.mean(normalized1) + np.mean(normalized2)),
    }


def compare_atomistic_energies(warm, random_result, N):
    """Compare raw feasible endpoints; a lower competitor need not be stationary."""
    comparison = {
        "N": N,
        f"{ATOMISTIC_RANDOM_SEED_TAG}_minus_warm_energy": np.nan,
        "energy_comparison_tolerance": np.nan,
        "lower_energy_competitor_found": None,
        "energy_comparison_status": "unavailable",
    }
    for result in (warm, random_result):
        if result is None or not np.isfinite(result["raw_energy"]):
            return comparison
        u = result["raw_u"]
        if u.size != 2 * N + 1 or not np.all(np.isfinite(u)):
            return comparison
        u1, u2 = split_layers(u, N)
        if abs(np.mean(u1) + np.mean(u2)) > MEAN_TOL:
            return comparison
    gap = random_result["raw_energy"] - warm["raw_energy"]
    tolerance = ENERGY_COMPARISON_RTOL * max(
        1.0, abs(warm["raw_energy"]), abs(random_result["raw_energy"]))
    lower_found = gap < -tolerance
    if lower_found:
        status = f"{ATOMISTIC_RANDOM_SEED_TAG}_lower"
    elif not (warm["accepted"] and random_result["accepted"]):
        status = "incomplete"
    else:
        status = "warm_lower" if gap > tolerance else "agree"
    comparison.update({
        f"{ATOMISTIC_RANDOM_SEED_TAG}_minus_warm_energy": float(gap),
        "energy_comparison_tolerance": float(tolerance),
        "lower_energy_competitor_found": bool(lower_found),
        "energy_comparison_status": status,
    })
    return comparison


def reflection_aligned_errors(u, reference, N):
    """Compare exact reflection representatives, retaining direct ties."""
    direct_errors = two_layer_errors(u, reference, N)
    reflected = reflected_state(u, N)
    reflected_errors = two_layer_errors(reflected, reference, N)
    improvement = direct_errors["h2_error"] - reflected_errors["h2_error"]
    threshold = REFLECTION_TIE_TOL * max(1.0, direct_errors["h2_error"])
    if improvement > threshold:
        return reflected, reflected_errors, True
    return u, direct_errors, False


def two_layer_errors(u, reference, N):
    """Return the paper's layer-weighted discrete H2 and maximum errors."""
    _, _, h1, h2, _ = atomistic_geometry(N)
    u1, u2 = split_layers(u, N)
    r1, r2 = split_layers(reference, N)
    error1, error2 = u1 - r1, u2 - r2
    l2 = np.sqrt(h1 * np.sum(error1**2) + h2 * np.sum(error2**2))
    gradient = np.sqrt(h1 * np.sum(((np.roll(error1, -1) - error1) / h1)**2) +
                       h2 * np.sum(((np.roll(error2, -1) - error2) / h2)**2))
    laplacian = np.sqrt(h1 * np.sum((
        (np.roll(error1, -1) - 2.0 * error1 + np.roll(error1, 1)) / h1**2
    )**2) + h2 * np.sum((
        (np.roll(error2, -1) - 2.0 * error2 + np.roll(error2, 1)) / h2**2)**2))
    maximum = max(float(np.max(np.abs(error1))), float(np.max(np.abs(error2))))
    return {
        "h2_error": float(np.sqrt(l2**2 + gradient**2 + laplacian**2)),
        "max_error": maximum,
    }


def loglog_slope(N, error):
    """Return the fitted slope of log(error) against log(N)."""
    slope = np.polyfit(np.log(np.asarray(N, dtype=float)),
                       np.log(np.asarray(error, dtype=float)), 1)[0]
    return float(slope)


def directional_gradient_error(value_gradient, point, direction, *args):
    """Return a centered directional-derivative relative error."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    _, gradient = value_gradient(point, *args)
    plus = value_gradient(point + GRADIENT_CHECK_STEP * direction, *args)[0]
    minus = value_gradient(point - GRADIENT_CHECK_STEP * direction, *args)[0]
    finite_difference = (plus - minus) / (2.0 * GRADIENT_CHECK_STEP)
    analytic = float(np.dot(gradient, direction))
    relative_error = abs(finite_difference - analytic) / max(
        1.0, abs(finite_difference), abs(analytic))
    return float(relative_error)


def reflected_state(u, N):
    """Apply the paper's odd reflection to both discrete layers."""
    u1, u2 = split_layers(u, N)
    return np.concatenate((
        -u1[(-np.arange(u1.size)) % u1.size],
        -u2[(-np.arange(u2.size)) % u2.size],
    ))


def local_verification_checks():
    """Run the focused construction and gradient checks."""
    rng = np.random.default_rng(123)
    N = 20
    point = 0.08 * rng.normal(size=2 * N + 1)
    reduced_point = atomistic_reduced_coordinates(point, N)
    reduced_direction = rng.normal(size=reduced_point.size)
    reduced_error = directional_gradient_error(atomistic_value_gradient,
                                               reduced_point,
                                               reduced_direction, N,
                                               PAIR_CUTOFF)

    continuum_x, continuum_dx = continuum_grid(N)
    continuum_point = 0.05 * rng.normal(size=N // 2 - 1)
    continuum_direction = rng.normal(size=continuum_point.size)
    continuum_error = directional_gradient_error(
        continuum_reduced_value_gradient, continuum_point, continuum_direction,
        continuum_x, continuum_dx, W, Wprime)

    random_reduced = random_fourier_atomistic_start(N)
    random_full = build_mean_gauge(random_reduced, N)
    random1, random2 = split_layers(random_full, N)
    random_energy, random_gradient = raw_atomistic_value_gradient(
        random_full, N, PAIR_CUTOFF)

    return {
        "reduced_gradient_relative_error":  #use the directional derivative to check whether the atomistic gradient is correct for N=20
        reduced_error,
        "continuum_gradient_relative_error":  # use the directional derivative to check whether the continuum gradient is correct for N=20
        continuum_error,
        "random_shape_ok":  #check whether the random Fourier start has the correct shape 2N
        bool(random_reduced.size == 2 * N),
        "random_gauge_defect":  #check the mean sum is 0
        float(abs(np.mean(random1) + np.mean(random2))),
        "random_initial_values_finite":  # check finiteness of energy and gradient for N=20
        bool(
            np.isfinite(random_energy) and np.all(np.isfinite(random_gradient))),
    }


def run_study():
    """Run the complete atomistic-to-continuum convergence study.

    Workflow:
        1. Check the building blocks. Run the inexpensive N=20 gradient
           and initial-validation checks.
        2. Compute the continuum reference. Solve the continuum problem on
           400- and 800-point meshes. The finer 800-point solution becomes the
           main reference.
        3. Run the atomistic convergence study. For every atom count N and
           both initial guesses, sample the continuum reference on the N and
            N+1 atomistic grids; run L-BFGS-B and at most one Newton correction;
            screen raw curvature, normalize, align its reflection representative, and
           calculate the discrete H_h^2 and maximum-norm errors.
        4. Measure convergence. Fit log-log slopes to determine whether the
           atomistic errors behave approximately like N^-1.
        5. Check numerical robustness. At the finest atomistic system, compare
           the M=80 and M=160 interaction cutoffs. Also compare the 400- and
           800-point continuum references.
        6. Assemble all validation results. Combine the local checks,
           optimizer endpoint checks, normalization checks, convergence
           slopes, monotonicity, cutoff sensitivity, and continuum-refinement
           checks.
        7. Return everything in one study dictionary containing solutions,
           errors, slopes, diagnostics, and PASS/FAIL checks.
    """
    verification = local_verification_checks()  #step 1

    # --------------------step 2------------------------#
    continuum_results = {
        points: solve_continuum_reference(points)
        for points in CONTINUUM_POINTS
    }
    for continuum in continuum_results.values():
        _, _, curvature = hessian_spectrum(continuum_full_hessian(
            continuum["q"], continuum["x"], continuum["dx"]))
        continuum.update(curvature)
    finest_continuum = continuum_results[CONTINUUM_POINTS[-1]]
    # For u1=q/2 and u2=-q/2, the two-layer continuum energy is F[q]/2.
    continuum_energy = 0.5 * finest_continuum["energy"]
    finest_spline = periodic_q_spline(finest_continuum["x"],
                                      finest_continuum["q"])

    # --------------------step 3------------------------#
    results = []
    energy_comparisons = []
    for N in N_VALUES:
        continuum_pair = sample_continuum_pair(
            finest_spline, N
        )  # sample the continuum reference on the N and N+1 atomistic grids
        #start the atomistic L-BFGS-B optimization for both initial guesses
        for start, initial in atomistic_initial_guesses(
                N, continuum_pair).items():
            #performs the optimization, applies Appendix-B normalization and computes endpoint diagnostics.
            run = solve_atomistic_initial(initial, N, PAIR_CUTOFF)
            #selects a odd symmetry-equivalent representative
            comparison_u, errors, reflection_applied = (
                reflection_aligned_errors(run["normalized_u"], continuum_pair,
                                          N))
            # Reduction singles out one gradient component, so its infinity norm
            # can change under reflection. Check the actual stored representative.
            comparison_diagnostics = atomistic_diagnostics(comparison_u, N, PAIR_CUTOFF)
            run.update(comparison_diagnostics)
            run["accepted"] = run["accepted"] and atomistic_stationary(comparison_diagnostics)
            #separates the displacement vector into the two layer arrays, length N and N+1, respectively
            mean1, mean2 = split_layers(comparison_u, N)
            run.update({
                "N": N,
                "h": atomistic_geometry(N)[-1],
                "start": start,
                "continuum_pair": continuum_pair,
                "normalized_u": comparison_u,
                "energy_minus_continuum": run["raw_energy"] - continuum_energy,
                "mean1": float(np.mean(mean1)),
                "mean2": float(np.mean(mean2)),
                "reflection_applied": reflection_applied,
                **errors,
            })
            # Phase motivation, using the final normalized/reflected endpoint:
            # m_N = mean(u_1,N) = -mean(u_2,N). For an unknown continuum phase
            # delta_*, write the layer-1 profile discrepancy as
            # u_1,N,n = u_1^c(x_1,n - delta_*) - delta_*/2 + epsilon_1,N,n.
            # Taking the discrete mean gives
            # m_N = -delta_*/2 + s_N(delta_*) + mean(epsilon_1,N),
            # s_N(delta_*) = (1/N) sum_n u_1^c(x_1,n - delta_*).
            # The continuous mean of u_1^c is zero, so s_N is the quadrature
            # discrepancy in its sampled mean. mean(epsilon_1,N) is the mean
            # profile discrepancy, not the atomistic EL consistency residual.
            # If both terms are small, delta_* is approximately -2*m_N.
            # Hence choose delta_N := -2*m_N = mean(u_2,N) - mean(u_1,N).
            # This selects a continuum reference; the matched norms below test
            # the remaining profile mismatch without changing the endpoint.
            phase_shift = run["mean2"] - run["mean1"]
            matched_errors = two_layer_errors(
                comparison_u, sample_continuum_pair(finest_spline, N, phase_shift), N)
            run.update({
                "continuum_phase_shift": phase_shift,
                "phase_matched_h2_error": matched_errors["h2_error"],
                "phase_matched_max_error": matched_errors["max_error"],
            })
            run["relative_mean_fraction"] = (4.0 * abs(run["mean1"]) /
                                             run["h"])
            run["h2_error_over_h"] = run["h2_error"] / run["h"]
            run["max_error_over_h"] = run["max_error"] / run["h"]
            results.append(run)
        cases = {result["start"]: result for result in results if result["N"] == N}
        comparison = compare_atomistic_energies(
            cases.get("sampled_continuum"), cases.get(ATOMISTIC_RANDOM_START_NAME), N)
        energy_comparisons.append(comparison)
        for result in cases.values():
            result.update(comparison)

    # --------------------step 4------------------------#
    slopes = {}
    for start in ATOMISTIC_START_NAMES:
        case = [result for result in results if result["start"] == start]
        if all(result["accepted"] for result in case):
            slopes[start] = {
                "h2":
                loglog_slope(N_VALUES,
                             [result["h2_error"] for result in case]),
                "max":
                loglog_slope(N_VALUES,
                             [result["max_error"] for result in case]),
                "phase_matched_h2": loglog_slope(
                    N_VALUES, [result["phase_matched_h2_error"] for result in case]),
                "phase_matched_max": loglog_slope(
                    N_VALUES, [result["phase_matched_max_error"] for result in case]),
            }
        else:
            slopes[start] = {"h2": np.nan, "max": np.nan,
                             "phase_matched_h2": np.nan, "phase_matched_max": np.nan}

    # --------------------step 5------------------------#
    #Atomistic cutoff \(M=80\) versus \(M=160\), checking the Euler-Lagrange residual over h.
    finest_N = N_VALUES[-1]
    cutoff_evaluations = []
    for result in (item for item in results if item["N"] == finest_N):
        _, gradient80 = raw_atomistic_value_gradient(result["normalized_u"],
                                                     finest_N, PAIR_CUTOFF)
        _, gradient160 = raw_atomistic_value_gradient(result["normalized_u"],
                                                      finest_N, CUTOFF_CHECK)
        difference1, difference2 = split_layers(gradient160 - gradient80,
                                                finest_N)
        _, _, h1, h2, h = atomistic_geometry(finest_N)
        el_difference = max(np.max(np.abs(difference1 / h1)),
                            np.max(np.abs(difference2 / h2)))
        cutoff_evaluations.append({
            "start":
            result["start"],
            "el_difference_inf_norm":
            float(el_difference),
            "el_difference_over_h":
            float(el_difference / h),
        })
    #Continuum reference on 400 versus 800 points, checking the H_h^2 and maximum errors.
    coarse_continuum = continuum_results[CONTINUUM_POINTS[0]]
    coarse_spline = periodic_q_spline(coarse_continuum["x"],
                                      coarse_continuum["q"])
    coarse_pair = sample_continuum_pair(coarse_spline, finest_N)
    fine_pair = next(result["continuum_pair"] for result in results
                     if result["N"] == finest_N)
    continuum_reference_change = two_layer_errors(coarse_pair, fine_pair,
                                                  finest_N)

    # --------------------step 6------------------------#
    warm = [
        result for result in results if result["start"] == "sampled_continuum"
    ]
    accepted_count = sum(result["accepted"] for result in results)
    maximum_mean_sum = max(
        abs(result["mean1"] + result["mean2"]) for result in results)
    # \max_{\text{all runs}}\left[\max(|\bar u_1|,|\bar u_2|)-\frac{h}{4}\right].\)
    maximum_bound_excess = max(
        max(abs(result["mean1"]), abs(result["mean2"])) -
        result["normalization"]["normalization_mean_bound"]
        for result in results)
    # measure how much changing the cutoff from \(M=80\) to \(M=160\) changes the EL residual, divided by \(h\)
    maximum_cutoff_difference = max(item["el_difference_over_h"]
                                    for item in cutoff_evaluations)

    checks = [
        ("mean-gauged atomistic gradient",
         verification["reduced_gradient_relative_error"] <= GRADIENT_CHECK_TOL,
         verification["reduced_gradient_relative_error"],
         f"<= {GRADIENT_CHECK_TOL:.1e}"),
        ("phase-fixed continuum gradient",
         verification["continuum_gradient_relative_error"]
         <= GRADIENT_CHECK_TOL,
         verification["continuum_gradient_relative_error"],
         f"<= {GRADIENT_CHECK_TOL:.1e}"),
        (f"seed-{ATOMISTIC_RANDOM_FOURIER_SEED} reduced length", verification["random_shape_ok"],
         float(verification["random_shape_ok"]), "== 1"),
        (f"seed-{ATOMISTIC_RANDOM_FOURIER_SEED} mean gauge", verification["random_gauge_defect"] <= MEAN_TOL,
         verification["random_gauge_defect"], f"<= {MEAN_TOL:.1e}"),
        (f"seed-{ATOMISTIC_RANDOM_FOURIER_SEED} finite energy and gradient",
         verification["random_initial_values_finite"],
         float(verification["random_initial_values_finite"]), "== 1"),
        ("accepted atomistic endpoints", accepted_count == len(results),
         float(accepted_count), f"== {len(results)}"),
        ("continuum curvature screen",
         all(item["curvature_status"] in ("soft", "positive")
             for item in continuum_results.values()),
         float(sum(item["curvature_status"] in ("soft", "positive")
                   for item in continuum_results.values())),
         f"== {len(continuum_results)}; lambda_min >= -tau_H"),
        ("two-start energy comparison",
         all(item["energy_comparison_status"] in ("agree", "warm_lower")
             for item in energy_comparisons),
         float(sum(item["energy_comparison_status"] in ("agree", "warm_lower")
                   for item in energy_comparisons)),
         f"== {len(N_VALUES)}; accepted pair and E_{ATOMISTIC_RANDOM_SEED_TAG} >= E_warm - tau_E"),
        ("Appendix B mean sums", maximum_mean_sum
         <= MEAN_TOL, maximum_mean_sum, f"<= {MEAN_TOL:.1e}"),
        ("Appendix B mean bounds", maximum_bound_excess
         <= 1.0e-14, maximum_bound_excess, "<= 1.0e-14"),
        ("M=80/160 EL difference over h", maximum_cutoff_difference
         <= CUTOFF_EL_OVER_H_TOL, maximum_cutoff_difference,
         f"<= {CUTOFF_EL_OVER_H_TOL:.1e}"),
        ("continuum 400/800 H_h^2 change",
         continuum_reference_change["h2_error"]
         < REFINEMENT_FRACTION * warm[-1]["h2_error"],
         continuum_reference_change["h2_error"],
         f"< 5% of {warm[-1]['h2_error']:.6e}"),
        ("continuum 400/800 maximum change",
         continuum_reference_change["max_error"]
         < REFINEMENT_FRACTION * warm[-1]["max_error"],
         continuum_reference_change["max_error"],
         f"< 5% of {warm[-1]['max_error']:.6e}"),
    ]
    for start, case_slopes in slopes.items():
        for norm in ("h2", "max"):
            slope = case_slopes[norm]
            checks.append(
                (f"{start} {norm} all-grid slope",
                 SLOPE_INTERVAL[0] <= slope <= SLOPE_INTERVAL[1], slope,
                 f"in [{SLOPE_INTERVAL[0]:.1f}, {SLOPE_INTERVAL[1]:.1f}]"
                 ))  #check the slope
    checks.extend([
        ("warm H_h^2 monotonic decrease",
         bool(np.all(np.diff([item["h2_error"] for item in warm]) < 0.0)),
         float(np.max(np.diff([item["h2_error"] for item in warm]))), "< 0"),
        ("warm maximum-error monotonic decrease",
         bool(np.all(np.diff([item["max_error"] for item in warm]) < 0.0)),
         float(np.max(np.diff([item["max_error"] for item in warm]))), "< 0"),
    ])
    #--------------------step 7------------------------#
    return {
        "verification": verification,
        "continuum_results": continuum_results,
        "results": results,
        "energy_comparisons": energy_comparisons,
        "slopes": slopes,
        "cutoff_evaluations": cutoff_evaluations,
        "continuum_reference_change": continuum_reference_change,
        "checks": checks,
    }


def write_csv(path, rows):
    """Write dictionaries to a CSV file with their first row's fixed schema."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_summary(outdir, study):
    """Save one fixed-schema row per atom count and start."""
    rows = []
    for result in study["results"]:
        normalization = result["normalization"]
        rows.append({
            "N":
            result["N"],
            "h":
            result["h"],
            "start":
            result["start"],
            "accepted":
            result["accepted"],
            "energy":
            result["energy"],
            "iterations":
            result["iterations"],
            "function_calls":
            result["function_calls"],
            "optimizer_task":
            result["optimizer_task"],
            "reduced_gradient_inf_norm":
            result["reduced_gradient_inf_norm"],
            "el_residual_inf_norm":
            result["el_residual_inf_norm"],
            "el_residual_over_h":
            result["el_residual_over_h"],
            "normalization_d":
            normalization["normalization_d"],
            "normalization_c":
            normalization["normalization_c"],
            "mean1":
            result["mean1"],
            "mean2":
            result["mean2"],
            "relative_mean_fraction":
            result["relative_mean_fraction"],
            "reflection_applied":
            result["reflection_applied"],
            "h2_error":
            result["h2_error"],
            "h2_error_over_h":
            result["h2_error_over_h"],
            "max_error":
            result["max_error"],
            "max_error_over_h":
            result["max_error_over_h"],
            **{key: result[key] for key in (
                "initial_energy", "initial_reduced_gradient_inf_norm",
                "lbfgsb_reduced_gradient_inf_norm", "raw_energy",
                "raw_reduced_gradient_inf_norm", "raw_el_residual_over_h",
                "gradient_reduction_ratio", "newton_correction_status",
                "newton_correction_applied", "newton_step_inf_norm",
                "curvature_min_eigenvalue", "curvature_tolerance", "curvature_status",
                f"{ATOMISTIC_RANDOM_SEED_TAG}_minus_warm_energy", "energy_comparison_tolerance",
                "lower_energy_competitor_found", "energy_comparison_status",
                "energy_minus_continuum", "continuum_phase_shift",
                "phase_matched_h2_error", "phase_matched_max_error")},
        })
    write_csv(outdir / "convergence_summary.csv", rows)


def save_profiles(outdir, study):
    """Save existing profile keys plus raw endpoints and continuum curvature."""
    arrays = {
        "N_values": np.asarray(N_VALUES),
        "pair_cutoff": np.asarray(PAIR_CUTOFF),
        "cutoff_check": np.asarray(CUTOFF_CHECK),
        "continuum_points": np.asarray(CONTINUUM_POINTS),
        "atomistic_start_names": np.asarray(ATOMISTIC_START_NAMES),
        "random_fourier_seed": np.asarray(ATOMISTIC_RANDOM_FOURIER_SEED),
    }
    for N in N_VALUES:
        _, _, h1, h2, _ = atomistic_geometry(N)
        cases = {
            result["start"]: result
            for result in study["results"] if result["N"] == N
        }
        warm1, warm2 = split_layers(cases["sampled_continuum"]["normalized_u"],
                                    N)
        cont1, cont2 = split_layers(
            cases["sampled_continuum"]["continuum_pair"], N)
        arrays.update({
            f"x1_N{N}": h1 * np.arange(N),
            f"x2_N{N}": h2 * np.arange(N + 1),
            f"u1_atomistic_N{N}": warm1,
            f"u2_atomistic_N{N}": warm2,
            f"u1_continuum_N{N}": cont1,
            f"u2_continuum_N{N}": cont2,
        })
        for start, result in cases.items():
            atom1, atom2 = split_layers(result["normalized_u"], N)
            arrays[f"u1_atomistic_{start}_N{N}"] = atom1
            arrays[f"u2_atomistic_{start}_N{N}"] = atom2
            arrays[f"raw_u_atomistic_{start}_N{N}"] = result["raw_u"]
    for points, continuum in study["continuum_results"].items():
        arrays[f"continuum_x_N{points}"] = continuum["x"]
        arrays[f"continuum_q_N{points}"] = continuum["q"]
        for key in ("curvature_min_eigenvalue", "curvature_tolerance", "curvature_status"):
            arrays[f"continuum_{key}_N{points}"] = np.asarray(continuum[key])
    np.savez(outdir / "profiles.npz", **arrays)


def save_plots(outdir, study):
    """Save the two-track H2 and maximum-error convergence figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean_mode_h2(result):
        """Return the H_h^2 contribution from the two layer-error means."""
        u1, u2 = split_layers(result["normalized_u"], result["N"])
        reference1, reference2 = split_layers(result["continuum_pair"],
                                              result["N"])
        mean_error1 = float(np.mean(u1 - reference1))
        mean_error2 = float(np.mean(u2 - reference2))
        return float(np.sqrt(TWOPI * (mean_error1**2 + mean_error2**2)))

    N = np.asarray(N_VALUES, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    styles = {
        "sampled_continuum": ("o-", "continuum warm start"),
        ATOMISTIC_RANDOM_START_NAME: (
            "s-", f"seed-{ATOMISTIC_RANDOM_FOURIER_SEED} phase-stress start"),
    }
    for start, (style, label) in styles.items():
        case = [
            result for result in study["results"] if result["start"] == start
        ]
        for axis, key, slope_key in ((axes[0], "h2_error", "h2"),
                                     (axes[1], "max_error", "max")):
            slope = study["slopes"][start][slope_key]
            slope_text = "unavailable" if not np.isfinite(
                slope) else f"{slope:.3f}"
            axis.loglog(N, [result[key] for result in case],
                        style,
                        linewidth=2,
                        label=f"{label} (slope={slope_text})")

    warm = [
        result for result in study["results"]
        if result["start"] == "sampled_continuum"
    ]
    warm_mean_modes = [mean_mode_h2(result) for result in warm]
    warm_mean_percentages = [
        100.0 * mean_mode / result["h2_error"]
        for mean_mode, result in zip(warm_mean_modes, warm)
    ]
    for axis, key in ((axes[0], "h2_error"), (axes[1], "max_error")):
        axis.loglog(N,
                    warm[0][key] * N[0] / N,
                    "k--",
                    alpha=0.65,
                    label="$N^{-1}$ guide")
    random_results = [
        result for result in study["results"]
        if result["start"] == ATOMISTIC_RANDOM_START_NAME
    ]
    random_mean_modes = [mean_mode_h2(result) for result in random_results]
    axes[0].loglog(N,
                   random_mean_modes,
                   ":",
                   color="tab:red",
                   linewidth=1.5,
                   label=(f"seed {ATOMISTIC_RANDOM_FOURIER_SEED} mean mode "
                          r"$2\sqrt{\pi}|\bar u_1|$"))
    for axis, norm in zip(axes, ("h2", "max")):
        slope = study["slopes"][ATOMISTIC_RANDOM_START_NAME][f"phase_matched_{norm}"]
        slope_text = "unavailable" if not np.isfinite(slope) else f"{slope:.3f}"
        axis.loglog(N,
                    [result[f"phase_matched_{norm}_error"] for result in random_results],
                    "--", color="tab:green", linewidth=1.5,
                    label=(f"seed {ATOMISTIC_RANDOM_FOURIER_SEED} vs shifted continuum "
                           f"(slope={slope_text})"))
    axes[0].annotate(
        ("warm mean mode\n" + r"$\leq$ "
         f"{max(warm_mean_percentages):.2f}% of total $H_h^2$ error"),
        xy=(N[0], warm[0]["h2_error"]),
        xytext=(5, -150),
        textcoords="offset points",
        fontsize=8,
        color="tab:blue",
        arrowprops={
            "arrowstyle": "->",
            "color": "tab:blue"
        },
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "tab:blue",
            "alpha": 0.9
        })
    axes[0].set_title(r"Layer-weighted $H_h^2$ error")
    axes[1].set_title("Maximum error")
    for axis in axes:
        axis.set_xlabel("number of layer-1 atoms $N$")
        axis.set_ylabel("error")
        axis.grid(True, which="both", linestyle="--", alpha=0.45)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Two atomistic solution tracks against the continuum reference")
    figure.tight_layout()
    figure.savefig(outdir / "convergence_loglog.png", dpi=180)
    plt.close(figure)


def save_report(outdir, study, overall_pass):
    """Write a compact report of the two tracks and necessary checks."""
    random_label = f"seed-{ATOMISTIC_RANDOM_FOURIER_SEED}"
    random_tag = ATOMISTIC_RANDOM_SEED_TAG
    warm = [
        result for result in study["results"]
        if result["start"] == "sampled_continuum"
    ]
    random_results = [
        result for result in study["results"]
        if result["start"] == ATOMISTIC_RANDOM_START_NAME
    ]
    failed = [check for check in study["checks"] if not check[1]]
    lines = [
        "Atomistic two-chain LJ convergence study",
        "=========================================",
        "",
        "Configuration",
        f"  N={N_VALUES}; continuum points={CONTINUUM_POINTS}",
        f"  atomistic cutoff M={PAIR_CUTOFF}; tail check M={CUTOFF_CHECK}",
        "  effective LJ: a=1, sigma=0.9, L=1, epsilon=0.5; W=4 sum V",
        (f"  starts=sampled_continuum, {ATOMISTIC_RANDOM_START_NAME}; seed="
         f"{ATOMISTIC_RANDOM_FOURIER_SEED}; modes=1-"
         f"{ATOMISTIC_RANDOM_FOURIER_MODES}; decay=1/k^2; amplitude="
         f"{ATOMISTIC_RANDOM_MAX_AMPLITUDE}"),
        (f"  acceptance: successful finite solve, Appendix B bounds, "
         f"gradient <= {ATOMISTIC_PGTOL:.1e}, EL_inf/h <= {ATOMISTIC_EL_OVER_H_TOL:.1e}, "
         "no resolved negative raw curvature"),
        "  gradients and EL are checked before and after normalization",
        "  one L-BFGS-B call; at most one conditional positive-spectrum Newton correction",
        (f"  Newton step <= {NEWTON_MAX_STEP_OVER_H:g}h; energy allowance = "
         f"{NEWTON_ENERGY_ROUNDOFF_FACTOR:g}*eps*max(1, |E_before|, |E_after|)"),
        (f"  tau_H = max({CURVATURE_SOFT_TOL:.1e}, "
         "100*eps*max(1, ||H||_inf) + 10*eigenpair residual)"),
        (f"  tau_E = {ENERGY_COMPARISON_RTOL:.1e}*max(1, |E_warm|, |E_{random_tag}|)"),
        "",
        f"Errors and {random_label} normalized mean",
        (f"  N    warm_H2     warm_max    {random_tag}_H2    {random_tag}_max   {random_tag}_mean1"
         f"  {random_tag}_warm_H2  {random_tag}_warm_max"),
    ]
    for warm_result, seed_result in zip(warm, random_results):
        cross_errors = two_layer_errors(seed_result["normalized_u"],
                                        warm_result["normalized_u"],
                                        warm_result["N"])
        lines.append(
            f"  {warm_result['N']:3d}  {warm_result['h2_error']:.4e}  "
            f"{warm_result['max_error']:.4e}  {seed_result['h2_error']:.4e}  "
            f"{seed_result['max_error']:.4e}  {seed_result['mean1']:+.4e}  "
            f"{cross_errors['h2_error']:13.4e}  {cross_errors['max_error']:14.4e}")
    lines.append(f"  {random_tag}_warm_*: {random_label} versus warm-start profiles after "
                 "normalization and reflection alignment.")
    lines.extend([
        "",
        f"{random_label.capitalize()} errors against a phase-matched continuum reference",
        (f"  {CONTINUUM_POINTS[-1]}-point reference; phase_shift = mean2 - mean1; "
         "atomistic endpoints unchanged"),
        "  N     phase_shift     matched_H2   matched_max",
    ])
    for result in random_results:
        lines.append(
            f"  {result['N']:3d}  {result['continuum_phase_shift']:+.6e}  "
            f"{result['phase_matched_h2_error']:.4e}  {result['phase_matched_max_error']:.4e}")
    matched_slopes = study["slopes"][ATOMISTIC_RANDOM_START_NAME]
    lines.extend([
        (f"  matched log-log slopes: H2={matched_slopes['phase_matched_h2']:.6f}, "
         f"max={matched_slopes['phase_matched_max']:.6f}"),
        "  green: phase-matched continuum reference; blue/orange: original fixed-phase reference",
    ])
    lines.extend([
        "",
        "Energy differences to continuum (E_atom - E_cont)",
        (f"  Finite-resolution reference: {CONTINUUM_POINTS[-1]} points; "
         "E_cont = F[q]/2 (two-layer energy)."),
        f"    N        warm_dE        {random_tag}_dE",
    ])
    for warm_result, seed_result in zip(warm, random_results):
        lines.append(
            f"  {warm_result['N']:3d}  {warm_result['energy_minus_continuum']:+.6e}  "
            f"{seed_result['energy_minus_continuum']:+.6e}")
    lines.extend([
        "",
        "All-grid log(error) versus log(N) slopes; expected -1",
        ("  sampled_continuum: H2="
         f"{study['slopes']['sampled_continuum']['h2']:.6f}, max="
         f"{study['slopes']['sampled_continuum']['max']:.6f}"),
        (f"  {ATOMISTIC_RANDOM_START_NAME}: H2="
         f"{study['slopes'][ATOMISTIC_RANDOM_START_NAME]['h2']:.6f}, max="
         f"{study['slopes'][ATOMISTIC_RANDOM_START_NAME]['max']:.6f}"),
        ("  largest scaled errors: H2/h="
         f"{max(result['h2_error_over_h'] for result in study['results']):.6f}, "
         f"max/h={max(result['max_error_over_h'] for result in study['results']):.6f}"
         ),
        (f"  {random_label} plateau ratio H2(640)/H2(320)="
         f"{random_results[-1]['h2_error'] / random_results[-2]['h2_error']:.6f}"),
    ])
    lines.extend(["", "Endpoint audit", f"  N    warm correction / curvature       {random_tag} correction / curvature      energy comparison"])
    for warm_result, seed_result, comparison in zip(warm, random_results, study["energy_comparisons"]):
        warm_status = (f"{warm_result['newton_correction_status']} / "
                       f"{warm_result['curvature_status']}")
        seed_status = (f"{seed_result['newton_correction_status']} / "
                       f"{seed_result['curvature_status']}")
        lines.append(f"  {warm_result['N']:3d}  {warm_status:<33} {seed_status:<33} "
                     f"{comparison['energy_comparison_status']}")
    for points, continuum in study["continuum_results"].items():
        lines.append(f"  continuum {points}: {continuum['curvature_status']} (full periodic space)")
    lines.extend([
        "  soft = unresolved near-zero curvature; no positive stability gap is claimed",
        "  agree = energies agree within tau_E for these two starts only",
        "  progress ratios (blank if initial gradient is zero) and eigenvalues are saved in CSV/NPZ",
    ])
    reference = study["continuum_results"][CONTINUUM_POINTS[-1]]
    spline = periodic_q_spline(reference["x"], reference["q"])
    lines.extend([
        "", "Continuum phase-mode overlap (raw endpoints; diagnostic only)",
        f"  {'N':>3} | {'start':<21} | {'lambda_min':>13} | {'lambda_2':>13} | "
        f"{'tau_H':>13} | continuum-phase overlap",
    ])
    for result in study["results"]:
        second, overlap = np.nan, np.nan
        if result["curvature_status"] != "unavailable":
            second, overlap = continuum_phase_mode_overlap(
                result["raw_u"], result["N"], spline, PAIR_CUTOFF)
        lines.append(
            f"  {result['N']:3d} | {result['start']:<21} | "
            f"{result['curvature_min_eigenvalue']:13.6e} | {second:13.6e} | "
            f"{result['curvature_tolerance']:13.6e} | {overlap:.8f}")
    lines.extend([
        "  overlap = absolute normalized dot product; near 1 means directions agree.",
        "  An isolated soft mode requires abs(lambda_min) <= tau_H < lambda_2.",
        "  For degenerate lowest modes, an individual eigenvector's overlap is not unique.",
        "  This is numerical evidence, not proof of exact atomistic symmetry or global minimality.",
        "  NaN = unavailable diagnostic; overlap does not affect acceptance or PASS/FAIL.",
        "", "Focused validation",
    ])
    check_labels = {
        "mean-gauged atomistic gradient":
        ("Atomistic directional-gradient accuracy", "Relative error {}"),
        "phase-fixed continuum gradient":
        ("Continuum directional-gradient accuracy", "Relative error {}"),
        f"{random_label} reduced length":
        (f"{random_label.capitalize()} initial vector length", "Correct length (2N)"),
        f"{random_label} mean gauge":
        (f"{random_label.capitalize()} initial layer means sum to zero", "abs(mean1 + mean2) {}"),
        f"{random_label} finite energy and gradient":
        (f"{random_label.capitalize()} initial energy and gradient are finite", "All values finite"),
        "accepted atomistic endpoints":
        ("All atomistic endpoints are accepted", "Accepted count {}"),
        "Appendix B mean sums":
        ("Appendix B layer means sum to zero", "abs(mean1 + mean2) {}"),
        "Appendix B mean bounds":
        ("Appendix B mean-bound excess", "Excess above h/4 {}"),
        "warm H_h^2 monotonic decrease":
        ("Warm H_h^2 error decreases on refinement",
         "Every consecutive error change {}"),
        "warm maximum-error monotonic decrease":
        ("Warm maximum error decreases on refinement",
         "Every consecutive error change {}"),
    }
    validation_rows = []
    for name, passed, value, requirement in study["checks"]:
        description = (name.replace("sampled_continuum", "warm")
                       .replace(ATOMISTIC_RANDOM_START_NAME, random_label)
                       .replace(" h2 ", " H_h^2 ")
                       .replace(" max ", " maximum-error "))
        description, criterion = check_labels.get(name, (description, "{}"))
        criterion = criterion.format(requirement)
        measured = (str(bool(value)) if name in (
            f"{random_label} reduced length", f"{random_label} finite energy and gradient")
                    else f"{value:.6e}")
        validation_rows.append((description, passed, measured, criterion))
    check_width = max([len("Check")] + [len(row[0]) for row in validation_rows])
    lines.extend([
        f"  {'Check':<{check_width}} | Result | Pass criterion",
        f"  {'-' * check_width}-+--------+---------------",
    ])
    for description, passed, _, criterion in validation_rows:
        mark = "\u221a" if passed else "\u00d7"
        lines.append(f"  {description:<{check_width}} | {mark:^6} | {criterion}")
    lines.append(f"  checks passed={len(study['checks']) - len(failed)}/"
                 f"{len(study['checks'])}")
    if failed:
        lines.extend(["", "  Failed check details"])
        for description, passed, measured, criterion in validation_rows:
            if not passed:
                lines.append(f"  FAIL {description}: measured={measured}, "
                             f"required {criterion}")
        for result in study["results"]:
            if not result["accepted"]:
                lines.append(
                    f"  endpoint N={result['N']} {result['start']}: "
                    f"raw gradient={result['raw_reduced_gradient_inf_norm']:.6e}, "
                    f"normalized gradient={result['reduced_gradient_inf_norm']:.6e}, "
                    f"raw EL/h={result['raw_el_residual_over_h']:.6e}, "
                    f"normalized EL/h={result['el_residual_over_h']:.6e}, "
                    f"lambda_min={result['curvature_min_eigenvalue']:.6e}, "
                    f"tau_H={result['curvature_tolerance']:.6e}; "
                    f"{result['optimizer_task']}; {result['newton_correction_status']}")
        for points, continuum in study["continuum_results"].items():
            if continuum["curvature_status"] not in ("soft", "positive"):
                lines.append(f"  continuum {points}: lambda_min="
                             f"{continuum['curvature_min_eigenvalue']:.6e}, "
                             f"tau_H={continuum['curvature_tolerance']:.6e}")
        for comparison in study["energy_comparisons"]:
            if comparison["energy_comparison_status"] not in ("agree", "warm_lower"):
                lines.append(f"  comparison N={comparison['N']}: "
                             f"{comparison['energy_comparison_status']}, "
                             f"E_{random_tag}-E_warm={comparison[f'{random_tag}_minus_warm_energy']:.6e}, "
                             f"tau_E={comparison['energy_comparison_tolerance']:.6e}")
    lines.extend([
        "",
        (f"Interpretation: seed {ATOMISTIC_RANDOM_FOURIER_SEED} is an intentionally selected illustrative "
         "phase-stress realization, not representative random-start statistics."
         ),
        ("Its finite-grid sawtooth is assessed by an O(h) envelope; smooth "
         "pointwise halving is not claimed."),
        ("Accepted endpoints are stationary candidates with no resolved negative "
         "curvature at the raw endpoint. Global minimality and the paper's "
         "stability-gap hypothesis are not verified."),
        f"OVERALL VERIFICATION CHECKS: {'PASS' if overall_pass else 'FAIL'}",
    ])
    report = "\n".join(lines) + "\n"
    (outdir / "check_report.txt").write_text(report, encoding="utf-8")
    # Keep Unicode in the file, and use aligned text on limited stdout encodings.
    try:
        report.encode(sys.stdout.encoding or "utf-8")
    except UnicodeEncodeError:
        report = (report.replace(f"| {chr(0x221a):^6} |", "|  PASS  |")
                  .replace(f"| {chr(0x00d7):^6} |", "|  FAIL  |"))
    print(report, end="")


def main():
    """Run the complete study, save its artifacts, and enforce verification."""
    study = run_study()
    overall_pass = all(passed for _, passed, _, _ in study["checks"])
    outdir = (Path(__file__).resolve().parents[2] / "outputs" / "relaxation" /
              "output_atomistic_two_chain_lj_lbfgs_elastic_dominance_convergence_study")
    outdir.mkdir(parents=True, exist_ok=True)
    save_summary(outdir, study)
    save_profiles(outdir, study)
    save_plots(outdir, study)
    save_report(outdir, study, overall_pass)
    if not overall_pass:
        raise RuntimeError(
            f"Numerical acceptance checks failed; see {outdir / 'check_report.txt'}"
        )


if __name__ == "__main__":
    main()
