"""Check Theorem 1 numerically without imposing oddness or phase.

Method:
    Forward periodic finite differences, deterministic unrestricted multistart,
    postprocessed phase alignment, mesh refinement, and L-BFGS-B optimization.
Purpose:
    Test whether unrestricted minimizers of Equation (6) become odd and
    strictly increasing after zero-crossing alignment, then compare them with
    a separate odd, phase-fixed reference calculation.
Inputs:
    N=100, 200, 400, alpha=1, and Phi(s)=cos(s), so W(s)=2*Phi(s).
Outputs:
    CSV diagnostics, NPZ profiles, two PNG comparisons, and a check report.
Output location:
    outputs/relaxation/output_jv_unrestricted_finite_difference_lbfgs_mesh_study.
Dependencies:
    NumPy, SciPy, Matplotlib, and clr.relaxation.odd_periodic.
Related files:
    jv_phase_fixed_finite_difference_lbfgs_single_case.py supplies the readable
    structural pattern and uses the same shared odd parametrization.
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np
import scipy.optimize as opt
from scipy.interpolate import CubicSpline

from clr.relaxation.odd_periodic import (
    build_odd_periodic as build_odd_u,
    reduce_odd_gradient as reduced_gradient,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MESH_SIZES = (100, 200, 400)
DENSE_POINTS = 4001
ALPHA = 1.0
RNG_SEED = 0

GRADIENT_TOL = 1e-7
EL_RESIDUAL_TOL = 1e-5
INITIAL_ODD_DEFECT_MIN = 1e-3
ENERGY_SPREAD_TOL = 2e-9
PROFILE_TOL = 1e-5
MAX_SOLVER_CALLS = 3


def grid(N):
    """Return the even periodic grid x_j=-pi+jh and spacing h."""
    if N % 2:
        raise ValueError("The odd reference parametrization requires even N.")
    return np.linspace(-np.pi, np.pi, N, endpoint=False), 2.0 * np.pi / N


def W(s):
    """Return W(s)=2*cos(s) for the paper's canonical Phi(s)=cos(s)."""
    return 2.0 * np.cos(s)


def Wprime(s):
    """Return the derivative W'(s)=-2*sin(s)."""
    return -2.0 * np.sin(s)


def energy_full(u, x, h, potential=W, potential_prime=Wprime):
    """Evaluate the transformed finite-difference energy E_h[u]."""
    du = np.roll(u, -1) - u
    return h * np.sum(0.5 * ALPHA * (du / h)**2 + potential(x + u))


def grad_full(u, x, h, potential=W, potential_prime=Wprime):
    """Return the gradient of E_h with respect to all periodic nodal values."""
    laplacian_part = ALPHA * (2.0 * u - np.roll(u, 1) - np.roll(u, -1)) / h
    return laplacian_part + h * potential_prime(x + u)


def energy_reduced(a, x, h, potential=W, potential_prime=Wprime):
    """Evaluate F(a)=E_h[build_odd_u(a)] for the reference solver."""
    return energy_full(build_odd_u(a, x.size), x, h, potential,
                       potential_prime)


def grad_reduced(a, x, h, potential=W, potential_prime=Wprime):
    """Return the reduced chain-rule gradient of F(a)."""
    return reduced_gradient(
        grad_full(build_odd_u(a, x.size), x, h, potential, potential_prime))


def reflection_defect(u):
    """Measure max_j |u(x_j)+u(-x_j)| using exact periodic grid indices."""
    reflected_indices = (-np.arange(u.size)) % u.size
    return float(np.max(np.abs(u + u[reflected_indices])))


def explicit_nonsymmetric_start(x):
    """Return the fixed smooth asymmetric Fourier initial condition."""
    return (0.31 + 0.27 * np.cos(x + 0.41) + 0.18 * np.sin(2.0 * x - 0.73) +
            0.09 * np.cos(3.0 * x + 0.22))


def unrestricted_initial_guesses(x):
    """Return one fixed and five seeded smooth nonsymmetric starts."""
    guesses = {"explicit_asymmetric": explicit_nonsymmetric_start(x)}
    rng = np.random.default_rng(RNG_SEED)
    modes = np.arange(1, 5, dtype=float)
    scales = 0.12 / modes

    for index in range(5):
        sine_coefficients = rng.normal(size=modes.size) * scales
        cosine_coefficients = rng.normal(size=modes.size) * scales
        guess = np.full(x.size, 0.15 if index % 2 == 0 else -0.15)
        for mode, sine_coefficient, cosine_coefficient in zip(
                modes, sine_coefficients, cosine_coefficients):
            guess += (sine_coefficient * np.sin(mode * x) +
                      cosine_coefficient * np.cos(mode * x))
        guesses[f"random_fourier_{index + 1}"] = guess

    return guesses


def reference_initial_guesses(x):
    """Return deterministic reduced-space starts for the odd reference."""
    x_free = x[1:x.size // 2]
    modes = np.arange(1, 5, dtype=float)
    coefficients = (np.random.default_rng(RNG_SEED).normal(size=modes.size) *
                    (0.12 / modes))
    random_odd = sum(coefficient * np.sin(mode * x_free)
                     for mode, coefficient in zip(modes, coefficients))
    return {
        "zero": np.zeros_like(x_free),
        "positive_sine": 0.5 * np.sin(x_free),
        "negative_sine": -0.5 * np.sin(x_free),
        "random_odd_fourier": random_odd,
    }


def solve_lbfgs(objective, gradient, initial, args,
                gradient_tol=GRADIENT_TOL):
    """Run safeguarded L-BFGS-B, restarting only a loose function stop."""
    point = np.array(initial, dtype=float, copy=True)
    total_iterations = 0

    for solver_call in range(1, MAX_SOLVER_CALLS + 1):
        point, energy, info = opt.fmin_l_bfgs_b(
            objective,
            point,
            fprime=gradient,
            args=args,
            pgtol=gradient_tol,
            factr=1.0,
            maxiter=5000,
            maxls=100,
        )
        total_iterations += int(info["nit"])
        gradient_inf = float(np.linalg.norm(info["grad"], ord=np.inf))
        task = str(info["task"])

        if (not np.isfinite(energy) or not np.all(np.isfinite(point))
                or not np.isfinite(gradient_inf)):
            raise RuntimeError(
                f"L-BFGS-B returned nonfinite data: task={task}")
        if info["warnflag"] != 0:
            raise RuntimeError(
                f"L-BFGS-B failed: task={task}, grad_inf={gradient_inf:.3e}")
        if gradient_inf <= gradient_tol:
            return {
                "point": point,
                "energy": float(energy),
                "gradient_inf_norm": gradient_inf,
                "task": task,
                "iterations": total_iterations,
                "solver_calls": solver_call,
            }
        if "RELATIVE REDUCTION OF F" not in task:
            raise RuntimeError(
                "L-BFGS-B stopped without meeting the gradient tolerance: "
                f"task={task}, grad_inf={gradient_inf:.3e}")

    raise RuntimeError(
        "L-BFGS-B restarts did not meet the gradient tolerance: "
        f"task={task}, grad_inf={gradient_inf:.3e}")


def solution_diagnostics(u, x, h, potential_prime=Wprime):
    """Return full Euler-Lagrange and forward-monotonicity diagnostics."""
    v = x + u
    residual = (ALPHA * (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / h**2 -
                potential_prime(v))
    forward_derivative = 1.0 + (np.roll(u, -1) - u) / h
    return {
        "el_residual_inf_norm": float(np.linalg.norm(residual, ord=np.inf)),
        "min_forward_derivative": float(np.min(forward_derivative)),
    }


def periodic_u_spline(x, u):
    """Return the periodic cubic interpolant of nodal correction values."""
    return CubicSpline(
        np.append(x, np.pi),
        np.append(u, u[0]),
        bc_type="periodic",
    )


def align_profile(u, x, dense_x, min_forward_derivative):
    """Apply the paper's zero-crossing normalization to the discrete profile."""
    # Paper notation: x=(x_j) is the optimization mesh, u=(u_h(x_j)) is the
    # optimized periodic correction, and the spline represents continuous u_h.
    spline = periodic_u_spline(x, u)  # u_h(x)

    # Since v_h(x)=x+u_h(x), sample v_h'(x)=1+u_h'(x) on dense_x.
    # This is checked together with the other two monotonicity measures below.
    initial_spline_min = float(np.min(1.0 + spline(dense_x, 1)))

    # Paper notation: left_value=v_h(-pi) and right_value=v_h(pi), where
    # v_h(x)=x+u_h(x) and v_h(pi)=v_h(-pi)+2*pi.
    left_value = float(-np.pi + spline(-np.pi))  # v_h(-pi)
    right_value = float(np.pi + spline(np.pi))  # v_h(pi)

    # Find the unique 2*pi*k in the half-open interval [v_h(-pi), v_h(pi));
    k = int(np.ceil(left_value / (2.0 * np.pi)))
    target = 2.0 * np.pi * k
    if not left_value <= target < right_value:
        raise RuntimeError(
            "The half-open zero-crossing target is not bracketed: "
            f"left={left_value:.16e}, target={target:.16e}, "
            f"right={right_value:.16e}")

    # Find a crossing candidate x_0 with v_h(x_0)=2*pi*k.  The combined
    # monotonicity check below certifies that this crossing is unique.
    zero_crossing_x = opt.brentq(
        lambda z: z + spline(z) - target,
        -np.pi,
        np.pi,
    )

    # Evaluate the paper's normalized profile v_0_h(x)=v_h(x+x_0)-2*pi*k on dense_x; hence v_0_h(0)=0.
    shifted_points = dense_x + zero_crossing_x  # Shift the dense grid by x_0.
    aligned = shifted_points + spline(
        shifted_points) - target  # v_0_h(shifted_points), s.t. v_0_h(0)=0.

    # Evaluate v_0_h(-x) and compute the oddness defect max_x |v_0_h(x)+v_0_h(-x)|.
    reflected_points = -dense_x + zero_crossing_x
    reflected = reflected_points + spline(reflected_points) - target

    # Check all three monotonicity measures together: the mesh forward
    # derivative, the spline derivative on dense_x, and the spline derivative
    # on the shifted dense grid used for v_0_h.
    shifted_spline_min = float(np.min(1.0 + spline(shifted_points, 1)))
    spline_min = min(initial_spline_min, shifted_spline_min)
    failed = (min_forward_derivative <= 0.0 or initial_spline_min <= 0.0
              or shifted_spline_min <= 0.0)
    if failed:
        return {
            "aligned_profile": None,
            "zero_crossing_x": float(zero_crossing_x),
            "aligned_oddness_defect": np.nan,
            "min_spline_derivative": spline_min,
            "alignment_pass": False,
        }

    return {
        "aligned_profile": aligned,
        "zero_crossing_x": float(zero_crossing_x),
        "aligned_oddness_defect": float(np.max(np.abs(aligned + reflected))),
        "min_spline_derivative": spline_min,
        "alignment_pass": True,
    }


def run_unrestricted_mesh(x, h, dense_x, potential=W,
                          potential_prime=Wprime,
                          gradient_tol=GRADIENT_TOL):
    """Run all unrestricted starts and return every stationary result."""
    runs = []
    for name, initial in unrestricted_initial_guesses(x).items():
        initial_odd_defect = reflection_defect(initial)
        if initial_odd_defect <= INITIAL_ODD_DEFECT_MIN:
            raise RuntimeError(
                f"Initial guess {name} is too close to the odd subspace: "
                f"defect={initial_odd_defect:.3e}")
        args = (x, h, potential, potential_prime)
        solved = solve_lbfgs(
            energy_full, grad_full, initial, args, gradient_tol)
        u = solved.pop("point")
        diagnostics = solution_diagnostics(u, x, h, potential_prime)
        alignment = align_profile(u, x, dense_x,
                                  diagnostics["min_forward_derivative"])
        runs.append({
            "start_name": name,
            "initial_odd_defect": initial_odd_defect,
            "u": u,
            "v": x + u,
            **solved,
            **diagnostics,
            **alignment,
        })
    return runs


def run_reference_mesh(x, h, dense_x, potential=W,
                       potential_prime=Wprime,
                       gradient_tol=GRADIENT_TOL):
    """Run the independent odd reduced-coordinate reference multistart."""
    runs = []
    for name, initial in reference_initial_guesses(x).items():
        args = (x, h, potential, potential_prime)
        solved = solve_lbfgs(
            energy_reduced, grad_reduced, initial, args, gradient_tol)
        a = solved.pop("point")
        u = build_odd_u(a, x.size)
        spline = periodic_u_spline(x, u)
        diagnostics = solution_diagnostics(u, x, h, potential_prime)
        runs.append({
            "start_name":
            name,
            "u":
            u,
            "v":
            x + u,
            "dense_profile":
            dense_x + spline(dense_x),
            "min_spline_derivative":
            float(np.min(1.0 + spline(dense_x, 1))),
            **solved,
            **diagnostics,
        })
    return runs


def run_mesh_study(potential=W, potential_prime=Wprime,
                   gradient_tol=GRADIENT_TOL,
                   el_residual_tol=EL_RESIDUAL_TOL):
    """Run all meshes and assemble comparisons and acceptance checks."""
    dense_x = np.linspace(-np.pi, np.pi, DENSE_POINTS)
    mesh_results = []

    for N in MESH_SIZES:
        x, h = grid(N)
        unrestricted_runs = run_unrestricted_mesh(
            x, h, dense_x, potential, potential_prime, gradient_tol)
        reference_runs = run_reference_mesh(
            x, h, dense_x, potential, potential_prime, gradient_tol)
        lowest_energy_unrestricted_run = min(
            unrestricted_runs, key=lambda run: run["energy"])
        lowest_energy_reference_run = min(
            reference_runs, key=lambda run: run["energy"])
        alignments_pass = all(run["alignment_pass"]
                              for run in unrestricted_runs)

        if alignments_pass:
            aligned_profiles = np.stack(
                [run["aligned_profile"] for run in unrestricted_runs])
            profile_spread = float(
                np.max(
                    np.abs(aligned_profiles -
                           lowest_energy_unrestricted_run["aligned_profile"])))
            max_oddness = max(run["aligned_oddness_defect"]
                              for run in unrestricted_runs)
            reference_difference = float(
                np.max(
                    np.abs(lowest_energy_unrestricted_run["aligned_profile"] -
                           lowest_energy_reference_run["dense_profile"])))
        else:
            aligned_profiles = np.full((len(unrestricted_runs), dense_x.size),
                                       np.nan)
            profile_spread = np.inf
            max_oddness = np.inf
            reference_difference = np.inf

        energies = [run["energy"] for run in unrestricted_runs]
        mesh_results.append({
            "N":
            N,
            "x":
            x,
            "h":
            h,
            "dense_x":
            dense_x,
            "unrestricted_runs":
            unrestricted_runs,
            "reference_runs":
            reference_runs,
            "lowest_energy_unrestricted_run":
            lowest_energy_unrestricted_run,
            "lowest_energy_reference_run":
            lowest_energy_reference_run,
            "aligned_profiles":
            aligned_profiles,
            "energy_spread":
            float(max(energies) - min(energies)),
            "profile_spread":
            profile_spread,
            "max_oddness_defect":
            float(max_oddness),
            "reference_energy_difference":
            abs(lowest_energy_unrestricted_run["energy"] -
                lowest_energy_reference_run["energy"]),
            "reference_profile_difference":
            reference_difference,
            "alignments_pass":
            alignments_pass,
            "mesh_profile_change":
            np.nan,
        })

    for previous, current in zip(mesh_results, mesh_results[1:]):
        if previous["alignments_pass"] and current["alignments_pass"]:
            current["mesh_profile_change"] = float(
                np.max(
                    np.abs(current["lowest_energy_unrestricted_run"]
                           ["aligned_profile"] -
                           previous["lowest_energy_unrestricted_run"]
                           ["aligned_profile"])))
        else:
            current["mesh_profile_change"] = np.inf

    finest = mesh_results[-1]
    finest_min_discrete = min(
        [run["min_forward_derivative"]
         for run in finest["unrestricted_runs"]] +
        [finest["lowest_energy_reference_run"]["min_forward_derivative"]])
    finest_min_spline = min(
        [run["min_spline_derivative"] for run in finest["unrestricted_runs"]] +
        [finest["lowest_energy_reference_run"]["min_spline_derivative"]])
    finest_residual = max(
        finest["lowest_energy_unrestricted_run"]["el_residual_inf_norm"],
        finest["lowest_energy_reference_run"]["el_residual_inf_norm"],
    )
    mesh_decrease = (np.isfinite(mesh_results[1]["mesh_profile_change"])
                     and np.isfinite(mesh_results[2]["mesh_profile_change"])
                     and (mesh_results[2]["mesh_profile_change"]
                          < mesh_results[1]["mesh_profile_change"]))
    mesh_change_ratio = (
        mesh_results[2]["mesh_profile_change"] /
        mesh_results[1]["mesh_profile_change"]
        if mesh_results[1]["mesh_profile_change"] > 0.0 else np.inf)
    all_alignments_pass = all(result["alignments_pass"]
                              for result in mesh_results)

    finest_mesh_label = f"Mesh size N={finest['N']} result"
    checks = [
        ("all unrestricted profiles alignable",
         all_alignments_pass, float(all_alignments_pass), "== 1"),
        (f"{finest_mesh_label}: Euler-Lagrange residual infinity norm",
         finest_residual <= el_residual_tol, finest_residual,
         f"<= {el_residual_tol:.1e}"),
        (f"{finest_mesh_label}: minimum forward derivative",
         finest_min_discrete > 0.0, finest_min_discrete, "> 0"),
        (f"{finest_mesh_label}: minimum cubic derivative",
         finest_min_spline > 0.0, finest_min_spline, "> 0"),
        (f"{finest_mesh_label}: unrestricted energy spread",
         finest["energy_spread"] <= ENERGY_SPREAD_TOL, finest["energy_spread"],
         f"<= {ENERGY_SPREAD_TOL:.1e}"),
        (f"{finest_mesh_label}: aligned profile spread",
         finest["profile_spread"] <= PROFILE_TOL, finest["profile_spread"],
         f"<= {PROFILE_TOL:.1e}"),
        (f"{finest_mesh_label}: maximum aligned oddness defect",
         finest["max_oddness_defect"] <= PROFILE_TOL,
         finest["max_oddness_defect"], f"<= {PROFILE_TOL:.1e}"),
        (f"{finest_mesh_label}: energy difference from odd reference",
         finest["reference_energy_difference"] <= ENERGY_SPREAD_TOL,
         finest["reference_energy_difference"], f"<= {ENERGY_SPREAD_TOL:.1e}"),
        (f"{finest_mesh_label}: profile difference from odd reference",
         finest["reference_profile_difference"] <= PROFILE_TOL,
         finest["reference_profile_difference"], f"<= {PROFILE_TOL:.1e}"),
        ("successive mesh-profile change decreases", mesh_decrease,
         mesh_change_ratio, "< 1"),
    ]
    return mesh_results, checks


def write_csv(path, rows, fieldnames):
    """Write dictionaries to a CSV file with a fixed, readable schema."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_mesh_summary(outdir, mesh_results, checks_pass):
    """Save one comparison and diagnostic row for each mesh."""
    rows = []
    for result in mesh_results:
        best = result["lowest_energy_unrestricted_run"]
        reference = result["lowest_energy_reference_run"]
        rows.append({
            "N":
            result["N"],
            "h":
            result["h"],
            "best_unrestricted_start":
            best["start_name"],
            "transformed_energy":
            best["energy"],
            "equation_6_energy":
            best["energy"] + ALPHA * np.pi,
            "gradient_inf_norm":
            best["gradient_inf_norm"],
            "el_residual_inf_norm":
            best["el_residual_inf_norm"],
            "min_forward_derivative":
            best["min_forward_derivative"],
            "min_spline_derivative":
            best["min_spline_derivative"],
            "zero_crossing_x":
            best["zero_crossing_x"],
            "best_aligned_oddness_defect":
            best["aligned_oddness_defect"],
            "max_aligned_oddness_defect":
            result["max_oddness_defect"],
            "unrestricted_energy_spread":
            result["energy_spread"],
            "aligned_profile_spread":
            result["profile_spread"],
            "odd_reference_energy":
            reference["energy"],
            "odd_reference_gradient_inf_norm":
            reference["gradient_inf_norm"],
            "odd_reference_el_residual_inf_norm":
            reference["el_residual_inf_norm"],
            "energy_difference_from_reference":
            result["reference_energy_difference"],
            "profile_difference_from_reference":
            result["reference_profile_difference"],
            "mesh_profile_change":
            result["mesh_profile_change"],
            "status":
            "PASS" if result["alignments_pass"] and
            (result is not mesh_results[-1] or checks_pass) else "FAIL",
        })
    write_csv(outdir / "mesh_summary.csv", rows, list(rows[0]))


def save_finest_multistart(outdir, finest):
    """Save every unrestricted result on the finest mesh."""
    rows = []
    for run in finest["unrestricted_runs"]:
        rows.append({
            "start_name": run["start_name"],
            "initial_odd_defect": run["initial_odd_defect"],
            "transformed_energy": run["energy"],
            "equation_6_energy": run["energy"] + ALPHA * np.pi,
            "gradient_inf_norm": run["gradient_inf_norm"],
            "el_residual_inf_norm": run["el_residual_inf_norm"],
            "min_forward_derivative": run["min_forward_derivative"],
            "min_spline_derivative": run["min_spline_derivative"],
            "zero_crossing_x": run["zero_crossing_x"],
            "aligned_oddness_defect": run["aligned_oddness_defect"],
            "optimizer_task": run["task"],
            "iterations": run["iterations"],
            "solver_calls": run["solver_calls"],
            "status": "PASS" if run["alignment_pass"] else "FAIL",
        })
    write_csv(outdir / "multistart_finest_mesh.csv", rows, list(rows[0]))


def save_profiles_and_plots(outdir, mesh_results):
    """Save the raw/aligned arrays and the two comparison figures."""
    finest = mesh_results[-1]
    best = finest["lowest_energy_unrestricted_run"]
    reference = finest["lowest_energy_reference_run"]
    np.savez(
        outdir / "finest_mesh_profiles.npz",
        x_nodes=finest["x"],
        u_unrestricted_raw_best=best["u"],
        v_unrestricted_raw_best=best["v"],
        x_dense=finest["dense_x"],
        unrestricted_start_names=np.array(
            [run["start_name"] for run in finest["unrestricted_runs"]]),
        v_unrestricted_aligned_all=finest["aligned_profiles"],
        v_unrestricted_aligned_best=(best["aligned_profile"] if
                                     best["aligned_profile"] is not None else
                                     np.full(finest["dense_x"].size, np.nan)),
        v_odd_reference=reference["dense_profile"],
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    for run in finest["unrestricted_runs"]:
        if run["aligned_profile"] is not None:
            axis.plot(finest["dense_x"],
                      run["aligned_profile"],
                      color="tab:blue",
                      alpha=0.25,
                      linewidth=1)
    if best["aligned_profile"] is not None:
        axis.plot(finest["dense_x"],
                  best["aligned_profile"],
                  color="tab:blue",
                  linewidth=2,
                  label="Unrestricted best")
    axis.plot(finest["dense_x"],
              reference["dense_profile"],
              "--",
              color="tab:orange",
              linewidth=2,
              label="Odd reference")
    axis.set(xlabel="$x$",
             ylabel="$v(x)$",
             title="Aligned unrestricted profiles and odd reference (N=400)")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(outdir / "aligned_profile_comparison.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for result in mesh_results:
        profile = result["lowest_energy_unrestricted_run"]["aligned_profile"]
        if profile is not None:
            axis.plot(result["dense_x"],
                      profile,
                      linewidth=2,
                      label=f"N={result['N']}")
    axis.set(xlabel="$x$",
             ylabel="$v(x)$",
             title="Mesh refinement of aligned unrestricted minimizers")
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(outdir / "mesh_refinement.png", dpi=150)
    plt.close(figure)


def save_report(outdir, mesh_results, checks, overall_pass,
                potential_description=None):
    """Write and print the complete numerical acceptance report."""
    lines = [
        "Independent numerical check of Theorem 1",
        "===========================================",
        "These computations are numerical evidence, not a proof of global "
        "minimality.",
    ]
    if potential_description is not None:
        lines.append(f"Potential: {potential_description}")
    lines.extend(["", "Mesh summaries"])
    for result in mesh_results:
        best = result["lowest_energy_unrestricted_run"]
        lines.extend([
            f"  N={result['N']}: best={best['start_name']}, "
            f"E={best['energy']:.15e}, J={best['energy'] + ALPHA*np.pi:.15e}",
            f"    grad_inf={best['gradient_inf_norm']:.3e}, "
            f"EL_residual={best['el_residual_inf_norm']:.3e}, "
            f"min_Dv={best['min_forward_derivative']:.6e}",
            f"    energy_spread={result['energy_spread']:.3e}, "
            f"profile_spread={result['profile_spread']:.3e}, "
            f"max_oddness={result['max_oddness_defect']:.3e}",
            f"    reference_dE={result['reference_energy_difference']:.3e}, "
            f"reference_dprofile={result['reference_profile_difference']:.3e}, "
            f"mesh_change={result['mesh_profile_change']:.3e}",
        ])

    lines.extend(["", "Acceptance checks"])
    for name, passed, value, requirement in checks:
        lines.append(f"  {'PASS' if passed else 'FAIL'}  {name}: "
                     f"value={value:.6e}, required {requirement}")
    lines.extend([
        "",
        f"OVERALL: {'PASS' if overall_pass else 'FAIL'}",
    ])
    report = "\n".join(lines) + "\n"
    (outdir / "check_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")


def run_and_save(potential, potential_prime, outdir,
                 potential_description=None, extra_checks=(),
                 gradient_tol=GRADIENT_TOL,
                 el_residual_tol=EL_RESIDUAL_TOL):
    """Run one potential study and save its complete diagnostic output."""
    mesh_results, checks = run_mesh_study(
        potential, potential_prime, gradient_tol, el_residual_tol)
    checks.extend(extra_checks)
    overall_pass = all(passed for _, passed, _, _ in checks)
    outdir.mkdir(parents=True, exist_ok=True)

    save_mesh_summary(outdir, mesh_results, overall_pass)
    save_finest_multistart(outdir, mesh_results[-1])
    save_profiles_and_plots(outdir, mesh_results)
    save_report(outdir, mesh_results, checks, overall_pass,
                potential_description)
    if not overall_pass:
        raise RuntimeError(
            f"Numerical acceptance criteria failed; see {outdir / 'check_report.txt'}"
        )
    return mesh_results, checks


def main():
    """Run the cosine study without changing its established output identity."""
    outdir = (Path(__file__).resolve().parents[2] / "outputs" / "relaxation" /
              "output_jv_unrestricted_finite_difference_lbfgs_mesh_study")
    run_and_save(W, Wprime, outdir)


if __name__ == "__main__":
    main()
