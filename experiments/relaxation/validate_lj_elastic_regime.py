"""Focused formula and saved-baseline checks for the LJ interaction sweep.

This script never launches a convergence study. It uses deterministic small
states for derivative and interval-bound checks and optionally compares two
explicitly supplied study pickles. Run only with trusted local pickle files.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.relaxation import atomistic_two_chain_lj_lbfgs_convergence_study as model
from experiments.relaxation.lj_elastic_dominance import (
    atomistic_regime,
    continuum_regime,
    pair_second_bounds,
)


def relative_difference(actual, expected):
    """Return maximum difference relative to max(1, maximum expected size)."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.max(np.abs(actual - expected)) /
                 max(1.0, float(np.max(np.abs(expected)))))


def directional_gradient_error(objective, state, direction):
    """Use a 1e-5 displacement to resolve energy differences at lambda=8."""
    direction = np.asarray(direction) / np.linalg.norm(direction)
    step = 1.0e-5
    measured = (objective(state + step * direction)[0]
                - objective(state - step * direction)[0]) / (2.0 * step)
    expected = float(objective(state)[1] @ direction)
    return abs(measured - expected) / max(1.0, abs(measured), abs(expected))


def pair_third_independent(s, parameters):
    """Differentiate the two radial powers directly, independently of the quartic."""
    c2 = (parameters.lattice_constant / (2.0 * np.pi))**2
    radius = parameters.interlayer_distance**2 + c2 * np.asarray(s)**2
    first, second = 2.0 * c2 * np.asarray(s), 2.0 * c2
    sigma6 = parameters.sigma**6
    return 4.0 * parameters.epsilon * (
        sigma6**2 * (-336.0 * first**3 / radius**9
                     + 126.0 * first * second / radius**8)
        + sigma6 * (60.0 * first**3 / radius**6
                    - 36.0 * first * second / radius**5))


def independent_stationary_points(parameters):
    """Bracket V''' zeros on a physical radial grid instead of solving its quartic."""
    radial_scale = 2.0 * np.pi * parameters.interlayer_distance / parameters.lattice_constant
    positive = radial_scale * np.geomspace(1.0e-7, 1.0e3, 12000)
    derivative = pair_third_independent(positive, parameters)
    roots = [0.0]
    for index in np.flatnonzero(derivative[:-1] * derivative[1:] < 0.0):
        root = brentq(pair_third_independent, positive[index],
                      positive[index + 1], args=(parameters,), xtol=1.0e-13)
        roots.extend((-root, root))
    return np.sort(roots)


def baseline_comparison(baseline, study):
    """Compare original numerical payload and endpoint statuses, ignoring new fields."""
    differences = []
    maximum_error = 0.0
    count = 0

    def compare(before, after, label):
        nonlocal maximum_error, count
        if isinstance(before, dict):
            for key, value in before.items():
                if key == "checks":
                    continue
                if key not in after:
                    differences.append(f"{label}.{key}: missing")
                else:
                    compare(value, after[key], f"{label}.{key}")
        elif isinstance(before, np.ndarray):
            count += 1
            if np.shape(before) != np.shape(after):
                differences.append(f"{label}: shape changed")
            elif not np.allclose(before, after, rtol=1.0e-9, atol=1.0e-11, equal_nan=True):
                differences.append(f"{label}: numeric array changed")
            if np.shape(before) == np.shape(after) and np.all(np.isfinite(before)):
                maximum_error = max(maximum_error, relative_difference(after, before))
        elif isinstance(before, (tuple, list)):
            if len(before) != len(after):
                differences.append(f"{label}: length changed")
            else:
                for index, (left, right) in enumerate(zip(before, after)):
                    compare(left, right, f"{label}[{index}]")
        elif isinstance(before, (bool, np.bool_, str)):
            count += 1
            if before != after:
                differences.append(f"{label}: status/text changed")
        elif isinstance(before, (float, int, np.number)):
            count += 1
            if not np.isclose(before, after, rtol=1.0e-9, atol=1.0e-11, equal_nan=True):
                differences.append(f"{label}: scalar changed")
            if np.isfinite(before) and np.isfinite(after):
                maximum_error = max(maximum_error, relative_difference(after, before))

    compare(baseline, study, "study")
    original_status = {item[0]: bool(item[1]) for item in baseline["checks"]}
    updated_status = {item[0]: bool(item[1])
                      for item in study.get("baseline_checks", study["checks"])}
    for name, passed in original_status.items():
        if updated_status.get(name) != passed:
            differences.append(f"check {name!r}: status changed or missing")
    return {
        "compared_values": count,
        "maximum_scaled_absolute_difference": maximum_error,
        "original_checks": len(original_status),
        "differences": differences,
    }


def run_checks(baseline_path=None, study_path=None):
    """Return descriptions, measured values, criteria, and outcomes for all checks."""
    checks = []

    def record(name, description, measured, criterion, passed):
        checks.append({"name": name, "description": description,
                       "measured": measured, "criterion": criterion,
                       "passed": None if passed is None else bool(passed),
                       "status": "SKIP" if passed is None else "PASS" if passed else "FAIL"})

    rng = np.random.default_rng(20916)
    N, cutoff = 20, model.PAIR_CUTOFF
    x, dx = model.continuum_grid(40)
    reduced_q = 0.025 * rng.normal(size=x.size // 2 - 1)
    q = model.build_odd_q(reduced_q, x.size)
    u = model.build_mean_gauge(model.random_fourier_atomistic_start(N), N)
    reduced_u = model.atomistic_reduced_coordinates(u, N)
    direction_q = rng.normal(size=q.size)
    direction_q /= np.linalg.norm(direction_q)
    direction_u = rng.normal(size=u.size)
    direction_u /= np.linalg.norm(direction_u)
    step = 1.0e-6

    for scale in (0.5, 1.0, 8.0):
        continuum = lambda state: model.continuum_full_value_gradient(
            state, x, dx, interaction_scale=scale)
        atomistic = lambda state: model.raw_atomistic_value_gradient(
            state, N, cutoff, interaction_scale=scale)
        for name, objective, state, direction, hessian in (
                ("continuum", continuum, q, direction_q,
                 model.continuum_full_hessian(q, x, dx, interaction_scale=scale)),
                ("atomistic", atomistic, u, direction_u,
                 model.raw_atomistic_hessian(u, N, cutoff, interaction_scale=scale))):
            gradient_error = directional_gradient_error(objective, state, direction)
            finite_hessian = (objective(state + step * direction)[1]
                              - objective(state - step * direction)[1]) / (2.0 * step)
            hessian_error = relative_difference(finite_hessian, hessian @ direction)
            record(f"{name}_derivatives_scale_{scale:g}",
                   "Central energy difference and gradient difference versus analytic derivatives.",
                   {"gradient_error": gradient_error, "hessian_error": hessian_error},
                   "Both scaled errors <= 1e-8", max(gradient_error, hessian_error) <= 1.0e-8)

        continuum_reduced_error = directional_gradient_error(
            lambda state: model.continuum_reduced_value_gradient(
                state, x, dx, interaction_scale=scale), reduced_q, rng.normal(size=reduced_q.size))
        atomistic_reduced_error = directional_gradient_error(
            lambda state: model.atomistic_value_gradient(state, N, cutoff, interaction_scale=scale),
            reduced_u, rng.normal(size=reduced_u.size))
        record(f"reduced_chain_rule_scale_{scale:g}",
               "Directional derivatives in the actual odd and mean-gauge solver coordinates.",
               {"continuum_error": continuum_reduced_error, "atomistic_error": atomistic_reduced_error},
               "Both scaled errors <= 1e-8",
               max(continuum_reduced_error, atomistic_reduced_error) <= 1.0e-8)

    _, _, h1, h2, _ = model.atomistic_geometry(N)
    elastic_hessian_u = np.zeros((u.size, u.size))
    elastic_hessian_u[:N, :N] = model.periodic_elastic_hessian(N, h1)
    elastic_hessian_u[N:, N:] = model.periodic_elastic_hessian(N + 1, h2)
    elastic_hessian_q = model.periodic_elastic_hessian(q.size, dx)
    for name, state, elastic_hessian, objective, hessian in (
            ("continuum", q, elastic_hessian_q,
             lambda scale: model.continuum_full_value_gradient(q, x, dx, interaction_scale=scale),
             lambda scale: model.continuum_full_hessian(q, x, dx, interaction_scale=scale)),
            ("atomistic", u, elastic_hessian_u,
             lambda scale: model.raw_atomistic_value_gradient(u, N, cutoff, interaction_scale=scale),
             lambda scale: model.raw_atomistic_hessian(u, N, cutoff, interaction_scale=scale))):
        elastic_gradient = elastic_hessian @ state
        elastic_energy = float(0.5 * state @ elastic_gradient)
        energy1, gradient1 = objective(1.0)
        errors = []
        for scale in (0.5, 8.0):
            energy, gradient = objective(scale)
            errors.extend((relative_difference(energy, elastic_energy + scale * (energy1 - elastic_energy)),
                           relative_difference(gradient, elastic_gradient + scale * (gradient1 - elastic_gradient)),
                           relative_difference(hessian(scale), elastic_hessian + scale * (hessian(1.0) - elastic_hessian))))
        record(f"{name}_interaction_linearity", "Only the nonlinear energy, gradient, and Hessian are scaled.",
               {"maximum_scaled_error": max(errors)}, "<= 5e-13", max(errors) <= 5.0e-13)

    custom_potential = lambda phase: np.cos(phase) + 0.3 * np.sin(2.0 * phase)
    custom_prime = lambda phase: -np.sin(phase) + 0.6 * np.cos(2.0 * phase)
    custom_energy, custom_gradient = model.continuum_full_value_gradient(
        q, x, dx, custom_potential, custom_prime, interaction_scale=8.0)
    expected_energy = 0.5 * q @ elastic_hessian_q @ q + 8.0 * dx * np.sum(custom_potential(x + q))
    expected_gradient = elastic_hessian_q @ q + 8.0 * dx * custom_prime(x + q)
    custom_error = max(relative_difference(custom_energy, expected_energy),
                       relative_difference(custom_gradient, expected_gradient))
    record("custom_potential_scaled_once", "Explicit continuum callback is multiplied by lambda exactly once.",
           {"maximum_scaled_error": custom_error}, "<= 5e-13", custom_error <= 5.0e-13)

    parameters = model.LJ_PARAMETERS
    stationary = independent_stationary_points(parameters)
    lower = np.r_[rng.uniform(-70.0, 20.0, 40), stationary, stationary - 1.0e-7, -1000.0, 0.0]
    upper = np.r_[lower[:40] + rng.uniform(0.001, 50.0, 40), stationary,
                  stationary + 1.0e-7, 1000.0, 0.0]
    bounded_lower, bounded_upper = pair_second_bounds(lower, upper, parameters)
    sampled_violation, exact_error = 0.0, 0.0
    for index, (left, right) in enumerate(zip(lower, upper)):
        candidates = np.r_[left, right, stationary[(stationary >= left) & (stationary <= right)]]
        values = model.Vsecond(candidates)
        exact_error = max(exact_error, abs(bounded_lower[index] - np.min(values)),
                          abs(bounded_upper[index] - np.max(values)))
        dense_values = model.Vsecond(np.linspace(left, right, 4097))
        sampled_violation = max(sampled_violation,
                                float(np.max(bounded_lower[index] - dense_values)),
                                float(np.max(dense_values - bounded_upper[index])))
    record("pair_second_interval_extrema",
           "Interval endpoints and independently bracketed V''' roots, with 4097 dense samples per interval.",
           {"maximum_extremum_difference": exact_error, "maximum_sample_violation": sampled_violation,
            "independent_stationary_points": stationary.tolist(), "interval_count": len(lower)},
           "Extremum difference <= 5e-13 and coverage violation <= 5e-14",
           exact_error <= 5.0e-13 and sampled_violation <= 5.0e-14)

    invalid_outcomes = {}
    for name, left, right in (("reversed", 1.0, -1.0), ("nan", np.nan, 1.0),
                              ("infinite", -np.inf, 1.0), ("shape", np.zeros(2), np.zeros(3))):
        try:
            pair_second_bounds(left, right, parameters)
            invalid_outcomes[name] = False
        except ValueError:
            invalid_outcomes[name] = True
    record("invalid_intervals", "Malformed intervals are rejected rather than reported as finite bounds.",
           invalid_outcomes, "Each case raises ValueError", all(invalid_outcomes.values()))

    reference = model.sample_continuum_pair(model.periodic_q_spline(x, q), N)
    mass_root = np.sqrt(np.r_[np.full(N, h1), np.full(N + 1, h2)])
    for scale in (0.5, 1.0, 8.0):
        regime = atomistic_regime(reference, u, N, parameters, interaction_scale=scale, cutoff=cutoff)
        bound = regime["force_lipschitz_bound"]
        norms = []
        for fraction in np.linspace(0.0, 1.0, 9):
            state = (1.0 - fraction) * reference + fraction * u
            interaction_hessian = model.raw_atomistic_hessian(
                state, N, cutoff, interaction_scale=scale) - elastic_hessian_u
            weighted_jacobian = interaction_hessian / mass_root[:, None] / mass_root[None, :]
            norms.append(float(np.max(np.abs(np.linalg.eigvalsh(weighted_jacobian)))))
        expected_cp = h1**2 / (4.0 * np.sin(h1 / 2.0)**2)
        cp_error = abs(regime["poincare_constant"] - expected_cp)
        record(f"weighted_segment_bound_scale_{scale:g}",
               "Sampled D_h^-1/2 H_interaction D_h^-1/2 spectral norms are below the whole-segment bound.",
               {"norms": norms, "bound": bound, "poincare_error": cp_error, "rho": regime["rho"]},
               "All 9 sampled norms <= bound + 1e-12; Poincare constant error <= 1e-14",
               max(norms) <= bound + 1.0e-12 and cp_error <= 1.0e-14)

    continuum80 = continuum_regime(parameters, cutoff=80, interval_count=2048)
    continuum_fine = continuum_regime(parameters, cutoff=80, interval_count=4096)
    continuum160 = continuum_regime(parameters, cutoff=160, interval_count=2048)
    phase = np.linspace(-np.pi, np.pi, 8193)
    sampled_second = 4.0 * np.sum(model.Vsecond(
        phase[:, None] - 2.0 * np.pi * np.arange(-80, 81)[None, :]), axis=1)
    dense_maximum = float(np.max(np.abs(sampled_second)))
    record("continuum_interval_refinement", "2048/4096 interval bounds both cover a dense independent image-sum sample.",
           {"rho_2048": continuum80["rho"], "rho_4096": continuum_fine["rho"], "dense_maximum": dense_maximum},
           "Dense maximum <= both bounds + 1e-12, and nested refinement does not enlarge bound",
           dense_maximum <= min(continuum80["rho"], continuum_fine["rho"]) + 1.0e-12
           and continuum_fine["rho"] <= continuum80["rho"] + 1.0e-12)
    segment80 = atomistic_regime(reference, u, N, parameters, cutoff=80)
    segment160 = atomistic_regime(reference, u, N, parameters, cutoff=160)
    differences = {"continuum_rho_difference": abs(continuum160["rho"] - continuum80["rho"]),
                   "atomistic_rho_difference": abs(segment160["rho"] - segment80["rho"])}
    record("regime_cutoff_sensitivity", "Finite-cutoff dominance diagnostics are stable between M=80 and M=160.",
           differences, "Both absolute rho differences <= 1e-6", max(differences.values()) <= 1.0e-6)

    if baseline_path is not None and study_path is not None:
        with Path(baseline_path).open("rb") as stream:
            baseline = pickle.load(stream)
        with Path(study_path).open("rb") as stream:
            study = pickle.load(stream)
        comparison = baseline_comparison(baseline, study)
        record("saved_lambda_one_baseline", "Original study payload and all original check outcomes under matched settings.",
               comparison, "Arrays/scalars rtol=1e-9, atol=1e-11; statuses and original checks unchanged",
               not comparison["differences"])
    else:
        record("saved_lambda_one_baseline", "Comparison requires both --baseline and --study; no study is launched here.",
               {"baseline": str(baseline_path), "study": str(study_path)}, "Supply both saved study paths", None)
    return {"checks": checks, "passed": all(item["passed"] is not False for item in checks),
            "performed_count": sum(item["passed"] is not None for item in checks),
            "skipped_count": sum(item["passed"] is None for item in checks),
            "python_executable": sys.executable,
            "limits": "Sampled verification checks implementation consistency; it is not a proof of exact-arithmetic bounds or global minimality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Trusted pre-change baseline_study.pkl")
    parser.add_argument("--study", type=Path, help="Trusted updated lambda=1 study pickle with matched settings")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    report = run_checks(args.baseline, args.study)
    args.outdir.mkdir(parents=True, exist_ok=True)
    report_path = args.outdir / "elastic_regime_validation.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{sum(item['passed'] is True for item in report['checks'])}/{report['performed_count']} checks passed; "
          f"{report['skipped_count']} skipped. Report: {report_path}")
    for item in report["checks"]:
        if item["passed"] is False:
            print(f"FAIL {item['name']}: {item['measured']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
