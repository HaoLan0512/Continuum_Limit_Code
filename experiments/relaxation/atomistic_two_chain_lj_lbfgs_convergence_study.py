"""Compare the paper's atomistic and continuum two-chain minimizers.

Method:
    Equations (12)-(13), a symmetric Lennard-Jones image cutoff, exact
    finite-cutoff gradients, phase-fixed continuum finite differences, and
    deterministic L-BFGS-B solves from a continuum warm start and one fixed
    smooth random-Fourier phase-stress start.
Purpose:
    Test the predicted O(N^-1) atomistic-to-continuum convergence rate in the
    layer-weighted discrete H^2 and maximum norms.
Inputs:
    N=20, 40, 80, 160, 320, 640; matched effective LJ parameters; M=80 images.
Outputs:
    One CSV diagnostic table, one NPZ profile archive, one PNG convergence
    figure, and a compact check report.
Output location:
    outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study.
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
ATOMISTIC_START_NAMES = ("sampled_continuum", "random_fourier_seed_3")
CONTINUUM_GRAD_TOL = 2.0e-7
CONTINUUM_EL_TOL = 2.0e-5
MAX_CONTINUUM_SOLVER_CALLS = 3
MAX_ITERATIONS = 5000
MAX_LINESEARCH = 100

GRADIENT_CHECK_STEP = 1.0e-6
GRADIENT_CHECK_TOL = 1.0e-8
SYMMETRY_TOL = 1.0e-10
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
    """Solve the continuum zero start, continuing at most twice after FACTR."""
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


def sample_continuum_pair(spline, N):
    """Sample u1=q/2 and u2=-q/2 on the two native atomistic grids. The q here is v_0(x)-x in the paper."""
    N1, N2, h1, h2, _ = atomistic_geometry(N)
    x1 = h1 * np.arange(N1)
    x2 = h2 * np.arange(N2)
    q1 = spline(wrap_to_paper_cell(x1))
    q2 = spline(wrap_to_paper_cell(x2))
    return np.concatenate((0.5 * q1, -0.5 * q2))


def random_fourier_atomistic_start(N):
    """Return the fixed seed-3 mean-gauged smooth phase-stress start.

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
        "random_fourier_seed_3": random_fourier_atomistic_start(N),
    }


def atomistic_diagnostics(u, N, cutoff):
    """Evaluate energy and stationarity once on a normalized endpoint."""
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


def solve_atomistic_initial(initial, N, cutoff):
    """Run one L-BFGS-B call and assess its Appendix-B-normalized endpoint."""
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
    endpoint = build_mean_gauge(point, N)
    if np.all(np.isfinite(endpoint)):
        normalized, normalization = appendix_b_normalize(endpoint, N)
        diagnostics = atomistic_diagnostics(normalized, N, cutoff)
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
    finite = (np.all(np.isfinite(point)) and np.all(np.isfinite(normalized))
              and all(np.isfinite(value) for value in diagnostics.values()))
    mean_ok = (abs(diagnostics["mean_sum"]) <= MEAN_TOL
               and max(abs(diagnostics["mean1"]), abs(diagnostics["mean2"]))
               <= normalization["normalization_mean_bound"] + 1.0e-14)
    accepted = (info["warnflag"] == 0 and finite and mean_ok and
                diagnostics["el_residual_over_h"] <= ATOMISTIC_EL_OVER_H_TOL)
    return {
        "normalized_u": normalized,
        "accepted": bool(accepted),
        "iterations": int(info["nit"]),
        "function_calls": int(info["funcalls"]),
        "optimizer_task": str(info["task"]),
        "warnflag": int(info["warnflag"]),
        "normalization": normalization,
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


def translated_state(u, N, d, c):
    """Apply the paper's exact infinite-sum discrete translation formula."""
    _, _, h1, h2, _ = atomistic_geometry(N)
    u1, u2 = split_layers(u, N)
    return np.concatenate((
        np.roll(u1, -d) + h2 * d + h2 * c,
        np.roll(u2, -d) + h1 * c,
    ))


def reflected_state(u, N):
    """Apply the paper's odd reflection to both discrete layers."""
    u1, u2 = split_layers(u, N)
    return np.concatenate((
        -u1[(-np.arange(u1.size)) % u1.size],
        -u2[(-np.arange(u2.size)) % u2.size],
    ))


def local_verification_checks():
    """Run the focused construction, gradient, and exact-symmetry checks."""
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

    gauged = project_mean_gauge(point, N)
    energy = raw_atomistic_value_gradient(gauged, N, PAIR_CUTOFF)[0]
    reflection_defect = abs(
        raw_atomistic_value_gradient(reflected_state(gauged, N), N,
                                     PAIR_CUTOFF)[0] - energy)
    translation_defect = abs(
        raw_atomistic_value_gradient(translated_state(gauged, N, 1, 0.0), N,
                                     PAIR_CUTOFF)[0] - energy)

    random_reduced = random_fourier_atomistic_start(N)
    repeated_random = random_fourier_atomistic_start(N)
    random_full = build_mean_gauge(random_reduced, N)
    random1, random2 = split_layers(random_full, N)
    random_energy, random_gradient = raw_atomistic_value_gradient(
        random_full, N, PAIR_CUTOFF)

    return {
        "reduced_gradient_relative_error":
        reduced_error,
        "continuum_gradient_relative_error":
        continuum_error,
        "symmetry_energy_defect":
        float(max(reflection_defect, translation_defect)),
        "random_reproducibility_defect":
        float(np.max(np.abs(random_reduced - repeated_random))),
        "random_shape_ok":
        bool(random_reduced.size == 2 * N),
        "random_gauge_defect":
        float(abs(np.mean(random1) + np.mean(random2))),
        "random_amplitude_defect":
        float(
            abs(np.max(np.abs(random_full)) - ATOMISTIC_RANDOM_MAX_AMPLITUDE)),
        "random_initial_values_finite":
        bool(
            np.isfinite(random_energy)
            and np.all(np.isfinite(random_gradient))),
    }


def run_study():
    """Compute two atomistic tracks, empirical rates, and focused checks."""
    verification = local_verification_checks()
    continuum_results = {
        points: solve_continuum_reference(points)
        for points in CONTINUUM_POINTS
    }
    finest_continuum = continuum_results[CONTINUUM_POINTS[-1]]
    finest_spline = periodic_q_spline(finest_continuum["x"],
                                      finest_continuum["q"])

    results = []
    for N in N_VALUES:
        continuum_pair = sample_continuum_pair(finest_spline, N)
        for start, initial in atomistic_initial_guesses(
                N, continuum_pair).items():
            run = solve_atomistic_initial(initial, N, PAIR_CUTOFF)
            comparison_u, errors, reflection_applied = (
                reflection_aligned_errors(run["normalized_u"], continuum_pair,
                                          N))
            mean1, mean2 = split_layers(comparison_u, N)
            run.update({
                "N": N,
                "h": atomistic_geometry(N)[-1],
                "start": start,
                "continuum_pair": continuum_pair,
                "normalized_u": comparison_u,
                "mean1": float(np.mean(mean1)),
                "mean2": float(np.mean(mean2)),
                "reflection_applied": reflection_applied,
                **errors,
            })
            run["relative_mean_fraction"] = (4.0 * abs(run["mean1"]) /
                                             run["h"])
            run["h2_error_over_h"] = run["h2_error"] / run["h"]
            run["max_error_over_h"] = run["max_error"] / run["h"]
            results.append(run)

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
            }
        else:
            slopes[start] = {"h2": np.nan, "max": np.nan}

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

    coarse_continuum = continuum_results[CONTINUUM_POINTS[0]]
    coarse_spline = periodic_q_spline(coarse_continuum["x"],
                                      coarse_continuum["q"])
    coarse_pair = sample_continuum_pair(coarse_spline, finest_N)
    fine_pair = next(result["continuum_pair"] for result in results
                     if result["N"] == finest_N)
    continuum_reference_change = two_layer_errors(coarse_pair, fine_pair,
                                                  finest_N)

    warm = [
        result for result in results if result["start"] == "sampled_continuum"
    ]
    seed3 = [
        result for result in results
        if result["start"] == "random_fourier_seed_3"
    ]
    accepted_count = sum(result["accepted"] for result in results)
    maximum_mean_sum = max(
        abs(result["mean1"] + result["mean2"]) for result in results)
    maximum_bound_excess = max(
        max(abs(result["mean1"]), abs(result["mean2"])) -
        result["normalization"]["normalization_mean_bound"]
        for result in results)
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
        ("exact atomistic symmetries", verification["symmetry_energy_defect"]
         <= SYMMETRY_TOL, verification["symmetry_energy_defect"],
         f"<= {SYMMETRY_TOL:.1e}"),
        ("seed-3 reproducibility",
         verification["random_reproducibility_defect"] == 0.0,
         verification["random_reproducibility_defect"], "== 0"),
        ("seed-3 reduced length", verification["random_shape_ok"],
         float(verification["random_shape_ok"]), "== 1"),
        ("seed-3 mean gauge", verification["random_gauge_defect"] <= MEAN_TOL,
         verification["random_gauge_defect"], f"<= {MEAN_TOL:.1e}"),
        ("seed-3 maximum amplitude", verification["random_amplitude_defect"]
         <= MEAN_TOL, verification["random_amplitude_defect"],
         f"<= {MEAN_TOL:.1e}"),
        ("seed-3 finite energy and gradient",
         verification["random_initial_values_finite"],
         float(verification["random_initial_values_finite"]), "== 1"),
        ("accepted atomistic endpoints", accepted_count == len(results),
         float(accepted_count), f"== {len(results)}"),
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
                 f"in [{SLOPE_INTERVAL[0]:.1f}, {SLOPE_INTERVAL[1]:.1f}]"))
    checks.extend([
        ("warm H_h^2 monotonic decrease",
         bool(np.all(np.diff([item["h2_error"] for item in warm]) < 0.0)),
         float(np.max(np.diff([item["h2_error"] for item in warm]))), "< 0"),
        ("warm maximum-error monotonic decrease",
         bool(np.all(np.diff([item["max_error"] for item in warm]) < 0.0)),
         float(np.max(np.diff([item["max_error"] for item in warm]))), "< 0"),
        ("seed-3 N=640 plateau break", seed3[-1]["h2_error"]
         < 0.5 * seed3[-2]["h2_error"],
         seed3[-1]["h2_error"] / seed3[-2]["h2_error"], "< 0.5"),
    ])

    return {
        "verification": verification,
        "continuum_results": continuum_results,
        "results": results,
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
        })
    write_csv(outdir / "convergence_summary.csv", rows)


def save_profiles(outdir, study):
    """Save continuum and both normalized atomistic tracks for every N."""
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
    for points, continuum in study["continuum_results"].items():
        arrays[f"continuum_x_N{points}"] = continuum["x"]
        arrays[f"continuum_q_N{points}"] = continuum["q"]
    np.savez(outdir / "profiles.npz", **arrays)


def save_plots(outdir, study):
    """Save the two-track H2 and maximum-error convergence figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.asarray(N_VALUES, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    styles = {
        "sampled_continuum": ("o-", "continuum warm start"),
        "random_fourier_seed_3": ("s-", "seed-3 phase-stress start"),
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
    for axis, key in ((axes[0], "h2_error"), (axes[1], "max_error")):
        axis.loglog(N,
                    warm[0][key] * N[0] / N,
                    "k--",
                    alpha=0.65,
                    label="$N^{-1}$ guide")
    seed3 = [
        result for result in study["results"]
        if result["start"] == "random_fourier_seed_3"
    ]
    axes[0].loglog(
        N, [2.0 * np.sqrt(PI) * abs(result["mean1"]) for result in seed3],
        ":",
        color="tab:red",
        linewidth=1.5,
        label=r"seed 3 mean mode $2\sqrt{\pi}|\bar u_1|$")
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
    warm = [
        result for result in study["results"]
        if result["start"] == "sampled_continuum"
    ]
    seed3 = [
        result for result in study["results"]
        if result["start"] == "random_fourier_seed_3"
    ]
    failed = [check for check in study["checks"] if not check[1]]
    lines = [
        "Atomistic two-chain LJ convergence study",
        "=========================================",
        "Two independent tracks are reported; neither is selected over the other.",
        "",
        "Configuration",
        f"  N={N_VALUES}; continuum points={CONTINUUM_POINTS}",
        f"  atomistic cutoff M={PAIR_CUTOFF}; tail check M={CUTOFF_CHECK}",
        "  effective LJ: a=1, sigma=0.9, L=1, epsilon=0.5; W=4 sum V",
        (f"  starts=sampled_continuum, random_fourier_seed_3; seed="
         f"{ATOMISTIC_RANDOM_FOURIER_SEED}; modes=1-"
         f"{ATOMISTIC_RANDOM_FOURIER_MODES}; decay=1/k^2; amplitude="
         f"{ATOMISTIC_RANDOM_MAX_AMPLITUDE}"),
        (f"  acceptance: successful finite solve, Appendix B bounds, "
         f"EL_inf/h <= {ATOMISTIC_EL_OVER_H_TOL:.1e}"),
        "",
        "Errors and seed-3 normalized mean",
        "  N    warm_H2     warm_max    seed3_H2    seed3_max   seed3_mean1",
    ]
    for warm_result, seed_result in zip(warm, seed3):
        lines.append(
            f"  {warm_result['N']:3d}  {warm_result['h2_error']:.4e}  "
            f"{warm_result['max_error']:.4e}  {seed_result['h2_error']:.4e}  "
            f"{seed_result['max_error']:.4e}  {seed_result['mean1']:+.4e}")
    lines.extend([
        "",
        "All-grid log(error) versus log(N) slopes; expected -1",
        ("  sampled_continuum: H2="
         f"{study['slopes']['sampled_continuum']['h2']:.6f}, max="
         f"{study['slopes']['sampled_continuum']['max']:.6f}"),
        ("  random_fourier_seed_3: H2="
         f"{study['slopes']['random_fourier_seed_3']['h2']:.6f}, max="
         f"{study['slopes']['random_fourier_seed_3']['max']:.6f}"),
        ("  largest scaled errors: H2/h="
         f"{max(result['h2_error_over_h'] for result in study['results']):.6f}, "
         f"max/h={max(result['max_error_over_h'] for result in study['results']):.6f}"
         ),
        ("  seed-3 plateau ratio H2(640)/H2(320)="
         f"{seed3[-1]['h2_error'] / seed3[-2]['h2_error']:.6f}"),
        "",
        "Focused validation",
        ("  directional errors: atomistic="
         f"{study['verification']['reduced_gradient_relative_error']:.3e}, "
         f"continuum={study['verification']['continuum_gradient_relative_error']:.3e}"
         ),
        ("  exact symmetry energy defect="
         f"{study['verification']['symmetry_energy_defect']:.3e}"),
        ("  continuum 400/800 change: H2="
         f"{study['continuum_reference_change']['h2_error']:.3e}, max="
         f"{study['continuum_reference_change']['max_error']:.3e}"),
        ("  largest M=80/160 EL difference over h="
         f"{max(item['el_difference_over_h'] for item in study['cutoff_evaluations']):.3e}"
         ),
        (f"  checks passed={len(study['checks']) - len(failed)}/"
         f"{len(study['checks'])}"),
    ])
    for name, _, value, requirement in failed:
        lines.append(f"  FAIL {name}: {value:.6e}, required {requirement}")
    lines.extend([
        "",
        ("Interpretation: seed 3 is an intentionally selected illustrative "
         "phase-stress realization, not representative random-start statistics."
         ),
        ("Its finite-grid sawtooth is assessed by an O(h) envelope; smooth "
         "pointwise halving is not claimed."),
        ("Accepted endpoints are local stationary candidates; global minimality "
         "and the paper's stability-gap hypothesis are not verified."),
        f"OVERALL VERIFICATION CHECKS: {'PASS' if overall_pass else 'FAIL'}",
    ])
    report = "\n".join(lines) + "\n"
    (outdir / "check_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")


def main():
    """Run the complete study, save its artifacts, and enforce verification."""
    study = run_study()
    overall_pass = all(passed for _, passed, _, _ in study["checks"])
    outdir = (Path(__file__).resolve().parents[2] / "outputs" / "relaxation" /
              "output_atomistic_two_chain_lj_lbfgs_convergence_study")
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
