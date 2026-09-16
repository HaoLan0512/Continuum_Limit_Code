"""Independent, small validation cases for warm-start safeguards.

This harness is intentionally outside the repository and never runs run_study.
"""
from pathlib import Path
import contextlib
import importlib.util
import json
import sys
from unittest.mock import patch

import numpy as np

REPO = Path(r"C:\Users\lh201\Desktop\Continuum_Limit_Relaxation\Continuum_Limit_Code")
SOURCE = REPO / "experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py"
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("warm_study_under_test", SOURCE)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)

RESULTS = []


def record(name, passed, measured=None, criterion=None):
    row = {"name": name, "passed": bool(passed)}
    if measured is not None:
        row["measured"] = measured
    if criterion is not None:
        row["criterion"] = criterion
    RESULTS.append(row)
    if not passed:
        raise AssertionError(row)


def compare(name, actual, expected, atol=1e-8, rtol=1e-8):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(actual - expected)))
    threshold = float(atol + rtol * np.max(np.abs(expected)))
    record(name, error <= threshold, error, f"absolute error <= {threshold:.3e}")


def identity_normalize(u, N):
    u1, u2 = study.split_layers(u, N)
    mean1, mean2 = float(u1.mean()), float(u2.mean())
    return u.copy(), {
        "normalization_d": 0,
        "normalization_c": 0.0,
        "normalization_mean_bound": study.atomistic_geometry(N)[-1] / 4,
        "mean1": mean1,
        "mean2": mean2,
        "mean_sum": mean1 + mean2,
    }


def optimizer_fixture(endpoint, N, gradient_fn, warnflag=0):
    """Return an actual reduced-state optimizer-shaped endpoint."""
    a = study.atomistic_reduced_coordinates(endpoint, N)
    energy, gradient = gradient_fn(study.build_mean_gauge(a, N), N, 2)
    info = {
        "warnflag": warnflag,
        "nit": 1,
        "funcalls": 2,
        "task": "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
        "grad": study.reduce_mean_gauge_gradient(gradient, N),
    }
    return a, energy, info


# Tests are appended after the source API has been finalized.

def test_hessians_and_gauge():
    rng = np.random.default_rng(20260914)
    for N in (1, 2, 5, 8):
        dimension = 2 * N + 1
        reflector = study.mean_gauge_reflector(N)
        Q = (np.eye(dimension) - 2 * np.outer(reflector, reflector))[:, :-1]
        normal = np.r_[np.full(N, 1 / N), np.full(N + 1, 1 / (N + 1))]
        normal /= np.linalg.norm(normal)
        compare(f"N={N} tangent columns orthonormal", Q.T @ Q, np.eye(2 * N), 2e-14, 0)
        compare(f"N={N} gauge normal orthogonal", normal @ Q, np.zeros(2 * N), 2e-14, 0)
        compare(f"N={N} tangent Euclidean projector", Q @ Q.T,
                np.eye(dimension) - np.outer(normal, normal), 2e-14, 0)
        u = Q @ rng.normal(scale=.02, size=2 * N)
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        for cutoff in (0, 3):
            H = study.raw_atomistic_hessian(u, N, cutoff)
            eps = 1e-6
            gp = study.raw_atomistic_value_gradient(u + eps * direction, N, cutoff)[1]
            gm = study.raw_atomistic_value_gradient(u - eps * direction, N, cutoff)[1]
            compare(f"N={N},M={cutoff} atomistic full Hessian FD", H @ direction,
                    (gp - gm) / (2 * eps), 5e-9, 2e-8)
            compare(f"N={N},M={cutoff} atomistic Hessian symmetry", H, H.T, 2e-14, 1e-14)
            HT, actual_reflector = study.atomistic_tangent_hessian(u, N, cutoff)
            compare(f"N={N},M={cutoff} tangent Hessian projection", HT, Q.T @ H @ Q, 2e-13, 1e-13)
            compare(f"N={N},M={cutoff} reflector returned", actual_reflector, reflector, 2e-14, 0)
    for points in (2, 8, 12):
        x, dx = study.continuum_grid(points)
        q = rng.normal(scale=.02, size=points)
        direction = rng.normal(size=points)
        direction /= np.linalg.norm(direction)
        for cutoff in (0, 3):
            H = study.continuum_full_hessian(q, x, dx, cutoff)
            wp = lambda s: study.Wprime(s, cutoff)
            w = lambda s: study.W(s, cutoff)
            eps = 1e-6
            gp = study.continuum_full_value_gradient(q + eps * direction, x, dx, w, wp)[1]
            gm = study.continuum_full_value_gradient(q - eps * direction, x, dx, w, wp)[1]
            compare(f"P={points},M={cutoff} continuum full Hessian FD", H @ direction,
                    (gp - gm) / (2 * eps), 5e-9, 2e-8)
            compare(f"P={points},M={cutoff} continuum Hessian symmetry", H, H.T, 2e-14, 1e-14)
    # The constant direction is excluded by the odd continuum coordinates.
    x, dx = study.continuum_grid(12)
    with patch.object(study, "Vsecond", lambda s: np.full_like(s, -.025)):
        H = study.continuum_full_hessian(np.zeros(12), x, dx, cutoff=0)
    B = np.column_stack([study.build_odd_q(a, 12) for a in np.eye(5)])
    record("continuum full-grid screen includes negative constant mode",
           np.linalg.eigvalsh(H)[0] < -1e-3 and np.linalg.eigvalsh(B.T @ H @ B)[0] > 0)


def test_curvature_classification():
    for name, diagonal, expected in (
        ("positive", [1., 2., 3.], "positive"),
        ("negative", [-1., 2., 3.], "negative"),
        ("zero", [0., 2., 3.], "soft"),
        ("small negative unresolved", [-1e-9, 2., 3.], "soft"),
    ):
        H = np.diag(diagonal)
        values, vectors, diagnostics = study.hessian_spectrum(H)
        record(f"curvature classification {name}", diagnostics["curvature_status"] == expected)
        compare(f"curvature eigenvalue {name}", values, diagonal, 2e-14, 0)
        residue = np.linalg.norm(H @ vectors[:, 0] - values[0] * vectors[:, 0])
        expected_tolerance = max(1e-8, 100 * np.finfo(float).eps * max(1., np.linalg.norm(H, np.inf)) + 10 * residue)
        compare(f"curvature tolerance formula {name}", diagnostics["curvature_tolerance"], expected_tolerance, 2e-15, 0)


def correction_case(name, coefficients, spectrum, hessian_factor=1., allow=True,
                    energy_penalty=False):
    N = 3
    dimension = 2 * N + 1
    reflector = study.mean_gauge_reflector(N)
    Q = (np.eye(dimension) - 2 * np.outer(reflector, reflector))[:, :-1]
    H = Q @ np.diag(spectrum) @ Q.T
    endpoint = Q @ np.asarray(coefficients)
    initial = endpoint.copy()

    def value_gradient(u, N, cutoff):
        value = -1.25 + .5 * u @ H @ u
        if energy_penalty and np.linalg.norm(u) < .5 * np.linalg.norm(initial):
            value += 1.
        return float(value), H @ u

    with patch.object(study, "raw_atomistic_value_gradient", value_gradient), \
            patch.object(study, "raw_atomistic_hessian", lambda u, N, cutoff: hessian_factor * H):
        returned, diagnostics, curvature, correction = study.polish_atomistic_endpoint(
            endpoint, N, 2, allow_correction=allow)
    return endpoint, returned, diagnostics, curvature, correction


def test_correction():
    zero = np.zeros(6)
    small = np.r_[1e-5, np.zeros(5)]
    positive = np.arange(1., 7.)
    initial, returned, diag, curv, corr = correction_case("stationary", zero, positive)
    record("already stationary skips correction", not corr["newton_correction_applied"])
    compare("already stationary endpoint retained", returned, initial, 0, 0)
    record("already stationary skips trial step", corr["newton_step_inf_norm"] == 0)

    initial, returned, diag, curv, corr = correction_case("accepted", small, positive)
    record("one Newton correction adopted", corr["newton_correction_applied"])
    record("adopted correction stationary", study.atomistic_stationary(diag))
    compare("adopted correction reaches quadratic minimum", returned, np.zeros(7), 2e-14, 0)

    for name, coeffs, spec, factor, allow, energy in (
        ("excessive step", np.r_[2., np.zeros(5)], positive, 1., True, False),
        ("unresolved stationarity", small, positive, 2., True, False),
        ("energy increase", small, positive, 1., True, True),
        ("negative curvature", small, np.r_[-1., np.arange(2., 7.)], 1., True, False),
        ("unsuccessful solver", small, positive, 1., False, False),
    ):
        initial, returned, diag, curv, corr = correction_case(name, coeffs, spec, factor, allow, energy)
        record(f"{name} rejects correction", not corr["newton_correction_applied"], corr["newton_correction_status"])
        compare(f"{name} preserves original endpoint", returned, initial, 0, 0)
        record(f"{name} original still not stationary", not study.atomistic_stationary(diag))
    # A soft component is deliberately left unchanged while positive modes improve.
    initial, returned, diag, curv, corr = correction_case("soft component", np.r_[.01, 1e-5, np.zeros(4)], np.r_[0., np.arange(2., 7.)])
    reflector = study.mean_gauge_reflector(3)
    Q = (np.eye(7) - 2 * np.outer(reflector, reflector))[:, :-1]
    compare("soft component unchanged by correction", (Q.T @ returned)[0], .01, 2e-14, 0)
    record("soft endpoint stationary without positive-gap claim", study.atomistic_stationary(diag) and curv["curvature_status"] == "soft")

def test_solve_integration():
    N = 3
    reflector = study.mean_gauge_reflector(N)
    Q = (np.eye(7) - 2 * np.outer(reflector, reflector))[:, :-1]
    H = Q @ np.diag(np.arange(1., 7.)) @ Q.T

    def value_gradient(u, N, cutoff):
        return float(-1.25 + .5 * u @ H @ u), H @ u

    def run_mock(endpoint, initial=None, warnflag=0, normalization=identity_normalize,
                 hessian=None):
        if initial is None:
            initial = Q[:, 0] * .02
        fixture = optimizer_fixture(endpoint, N, value_gradient, warnflag)
        with patch.object(study, "raw_atomistic_value_gradient", value_gradient), \
                patch.object(study, "raw_atomistic_hessian", hessian or (lambda u, N, cutoff: H)), \
                patch.object(study, "appendix_b_normalize", normalization), \
                patch.object(study.opt, "fmin_l_bfgs_b", return_value=fixture) as optimizer:
            result = study.solve_atomistic_initial(study.atomistic_reduced_coordinates(initial, N), N, 2)
        record("solve retains exactly one L-BFGS-B call", optimizer.call_count == 1)
        return result

    result = run_mock(np.zeros(7))
    record("stationary stable successful solve accepted", result["accepted"])
    record("stationary stable successful solve skips correction", not result["newton_correction_applied"])
    compare("pre-normalization initial energy recorded", result["initial_energy"], value_gradient(Q[:, 0] * .02, N, 2)[0], 2e-14, 0)
    compare("pre-normalization final energy recorded", result["raw_energy"], -1.25, 2e-14, 0)
    compare("gradient reduction recorded", result["gradient_reduction_ratio"], 0., 2e-14, 0)
    result = run_mock(np.zeros(7), initial=np.zeros(7))
    ratio = result["gradient_reduction_ratio"]
    record("zero initial gradient reduction is not applicable", ratio is None or np.isnan(ratio))

    result = run_mock(Q[:, 0] * 1e-5)
    record("solver stagnation corrected once and accepted", result["newton_correction_applied"] and result["accepted"])
    record("solver gradient and final raw gradient distinguished",
           result["lbfgsb_reduced_gradient_inf_norm"] > study.ATOMISTIC_PGTOL and
           result["raw_reduced_gradient_inf_norm"] <= study.ATOMISTIC_PGTOL)
    result = run_mock(Q[:, 0] * 1e-5, warnflag=1)
    record("unsuccessful solver cannot use correction or be accepted", not result["accepted"] and not result["newton_correction_applied"])

    def shifted_normalize(u, N):
        return identity_normalize(u + Q[:, 0] * .001, N)

    result = run_mock(np.zeros(7), normalization=shifted_normalize)
    record("normalized-output stationarity independently enforced",
           result["raw_reduced_gradient_inf_norm"] <= study.ATOMISTIC_PGTOL and
           result["reduced_gradient_inf_norm"] > study.ATOMISTIC_PGTOL and not result["accepted"])

    hessian_calls = []
    def changing_hessian(u, N, cutoff):
        hessian_calls.append(u.copy())
        return H if np.linalg.norm(u) > 1e-8 else H - 2 * np.outer(Q[:, 0], Q[:, 0])

    result = run_mock(Q[:, 0] * 1e-5, hessian=changing_hessian)
    record("curvature recomputed after correction", len(hessian_calls) >= 2)
    record("newly negative curvature blocks accepted endpoint", not result["accepted"] and result["curvature_status"] == "negative")


def test_energy_comparison():
    N = 3
    u = np.zeros(2 * N + 1)
    def endpoint(energy, accepted=True, state=None):
        return {
            "N": N, "raw_u": u.copy() if state is None else state,
            "raw_energy": energy, "accepted": accepted, "warnflag": 0 if accepted else 1,
            "raw_reduced_gradient_inf_norm": 0. if accepted else 1.,
            "raw_el_residual_over_h": 0. if accepted else 1.,
        }
    for name, warm, seed3, expected, found in (
        ("agreement", endpoint(-1.), endpoint(-1. + 1e-12), "agree", False),
        ("warm lower", endpoint(-1.1), endpoint(-1.), "warm_lower", False),
        ("seed3 lower", endpoint(-1.), endpoint(-1.1), "seed3_lower", True),
        ("failed feasible lower competitor", endpoint(-1.), endpoint(-1.1, False), "seed3_lower", True),
        ("failed nonlower competitor", endpoint(-1.), endpoint(-1., False), "incomplete", False),
        ("failed warm with nonlower competitor", endpoint(-1., False), endpoint(-1.), "incomplete", False),
        ("missing competitor", endpoint(-1.), None, "unavailable", None),
        ("missing warm", None, endpoint(-1.), "unavailable", None),
        ("nonfinite energy", endpoint(-1.), endpoint(np.nan), "unavailable", None),
        ("nonfinite raw endpoint", endpoint(-1.), endpoint(-2., state=np.full(7, np.nan)), "unavailable", None),
        ("infeasible lower endpoint", endpoint(-1.), endpoint(-2., state=np.ones(7)), "unavailable", None),
    ):
        result = study.compare_atomistic_energies(warm, seed3, N)
        record(f"energy comparison {name}", result["energy_comparison_status"] == expected,
               result["energy_comparison_status"])
        record(f"energy comparison competitor flag {name}", result["lower_energy_competitor_found"] is found,
               result["lower_energy_competitor_found"])
        if expected not in ("unavailable", "incomplete"):
            expected_tolerance = 1e-10 * max(1., abs(warm["raw_energy"]), abs(seed3["raw_energy"]))
            compare(f"energy comparison tolerance {name}", result["energy_comparison_tolerance"], expected_tolerance, 2e-20, 0)



def test_cached_study_integration():
    """Replay saved real endpoints through final aggregation; prohibit optimization."""
    import copy
    import csv
    import hashlib
    import io
    import pickle
    import shutil
    import warnings

    out = REPO / "outputs/relaxation/warm_start_validation_20260914_122932"
    if not (out / "profiles.npz").exists():
        raise FileNotFoundError("Cached full-study outputs are required for integration validation")
    baseline_files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (out / "baseline").rglob("*") if p.is_file()}
    with (out / "convergence_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with np.load(out / "profiles.npz") as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    text_keys = {"start", "optimizer_task", "newton_correction_status", "curvature_status", "energy_comparison_status"}
    bool_keys = {"accepted", "reflection_applied", "newton_correction_applied", "lower_energy_competitor_found"}
    int_keys = {"N", "iterations", "function_calls", "normalization_d"}
    cached = {}
    for row in rows:
        result = {}
        for key, value in row.items():
            if key in text_keys:
                result[key] = value
            elif key in bool_keys:
                result[key] = value == "True" if value else None
            elif key in int_keys:
                result[key] = int(value)
            else:
                result[key] = float(value) if value else None
        N, start = result["N"], result["start"]
        raw = arrays[f"raw_u_atomistic_{start}_N{N}"]
        normalized, normalization = study.appendix_b_normalize(raw, N)
        result.update(raw_u=raw, normalized_u=normalized, normalization=normalization,
                      warnflag=0, cached_endpoint_replay=True)
        cached[N, start] = result

    continuum_cache = {}
    for points in study.CONTINUUM_POINTS:
        x, q = arrays[f"continuum_x_N{points}"], arrays[f"continuum_q_N{points}"]
        dx = study.TWOPI / points
        energy, gradient = study.continuum_full_value_gradient(q, x, dx)
        continuum_cache[points] = {
            "point_count": points, "x": x, "q": q, "dx": dx,
            "energy": energy,
            "gradient_inf_norm": float(np.max(np.abs(study.reduce_odd_gradient(gradient)))),
            "cached_endpoint_replay": True,
            **study.continuum_diagnostics(q, x, dx),
        }

    def replay(failure_fixture=False):
        atomistic_counts = {}
        original_diagnostics = study.atomistic_diagnostics
        original_reflection = study.reflection_aligned_errors
        reflected_failure_state = []

        def cached_atomistic(initial, N, cutoff):
            index = atomistic_counts.get(N, 0)
            atomistic_counts[N] = index + 1
            if index >= len(study.ATOMISTIC_START_NAMES):
                raise AssertionError("Unexpected extra atomistic solve")
            return copy.deepcopy(cached[N, study.ATOMISTIC_START_NAMES[index]])

        def controlled_reflection(u, reference, N):
            comparison_u, errors, reflected = original_reflection(u, reference, N)
            if failure_fixture and N == study.N_VALUES[0] and not reflected_failure_state:
                comparison_u = study.reflected_state(u, N)
                errors = study.two_layer_errors(comparison_u, reference, N)
                reflected_failure_state.append(comparison_u)
                return comparison_u, errors, True
            return comparison_u, errors, reflected

        def controlled_diagnostics(u, N, cutoff):
            diagnostics = original_diagnostics(u, N, cutoff)
            if failure_fixture and reflected_failure_state and u is reflected_failure_state[0]:
                diagnostics["reduced_gradient_inf_norm"] = 2 * study.ATOMISTIC_PGTOL
            return diagnostics

        with patch.object(study, "solve_continuum_reference", side_effect=lambda points: copy.deepcopy(continuum_cache[points])) as continuum_solver, \
                patch.object(study, "solve_atomistic_initial", side_effect=cached_atomistic) as atomistic_solver, \
                patch.object(study.opt, "fmin_l_bfgs_b", side_effect=AssertionError("Optimizer execution forbidden during cached replay")) as optimizer, \
                patch.object(study, "reflection_aligned_errors", side_effect=controlled_reflection), \
                patch.object(study, "atomistic_diagnostics", side_effect=controlled_diagnostics):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = study.run_study()
        record("cached replay executes no optimizer", optimizer.call_count == 0)
        record("cached replay uses two continuum and twelve atomistic endpoints", continuum_solver.call_count == 2 and atomistic_solver.call_count == 12)
        record("cached replay raises no warnings", len(caught) == 0, [str(w.message) for w in caught])
        return result

    final = replay()
    record("cached full study retains all nineteen acceptance checks", len(final["checks"]) == 19 and all(ok for _, ok, _, _ in final["checks"]))
    record("cached full study accepts all twelve endpoints", len(final["results"]) == 12 and all(r["accepted"] for r in final["results"]))
    reflection_count = sum(r["reflection_applied"] for r in final["results"])
    record("cached full study preserves reflection decisions", reflection_count == sum(row["reflection_applied"] == "True" for row in rows), reflection_count)
    for result in final["results"]:
        N, start = result["N"], result["start"]
        original = np.r_[arrays[f"u1_atomistic_{start}_N{N}"], arrays[f"u2_atomistic_{start}_N{N}"]]
        compare(f"cached {N}/{start} saved representative preserved", result["normalized_u"], original, 0, 0)
        current_diag = study.atomistic_diagnostics(result["normalized_u"], N, study.PAIR_CUTOFF)
        compare(f"cached {N}/{start} saved gradient matches actual representative", result["reduced_gradient_inf_norm"], current_diag["reduced_gradient_inf_norm"], 0, 0)
        record(f"cached {N}/{start} actual representative stationary", study.atomistic_stationary(current_diag))

    failed = replay(failure_fixture=True)
    bad = failed["results"][0]
    record("controlled postreflection gradient failure rejects endpoint", bad["reflection_applied"] and not bad["accepted"] and bad["reduced_gradient_inf_norm"] > study.ATOMISTIC_PGTOL)
    record("controlled postreflection failure fails existing acceptance check", not next(ok for name, ok, _, _ in failed["checks"] if name == "accepted atomistic endpoints"))
    record("controlled postreflection failure withholds track slopes", all(np.isnan(v) for v in failed["slopes"]["sampled_continuum"].values()))
    record("controlled postreflection failure makes comparison incomplete", failed["energy_comparisons"][0]["energy_comparison_status"] == "incomplete")
    record("controlled postreflection failure preserves independent track", all(r["accepted"] for r in failed["results"] if r["start"] == "random_fourier_seed_3"))

    # Refresh only this task's artifacts from the successful cached replay.
    study.save_summary(out, final)
    study.save_profiles(out, final)
    study.save_plots(out, final)
    with contextlib.redirect_stdout(io.StringIO()):
        study.save_report(out, final, True)
    report = (out / "check_report.txt").read_text(encoding="utf-8")
    record("final report includes Newton and curvature/energy tolerances", all(token in report for token in ("Newton step <= 0.1h", "tau_H =", "tau_E =", "OVERALL VERIFICATION CHECKS: PASS")))
    with np.load(out / "profiles.npz") as archive:
        record("cached replay preserves every NPZ array", set(archive.files) == set(arrays) and all(np.array_equal(archive[k], v) for k, v in arrays.items()))
    baseline_after = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (out / "baseline").rglob("*") if p.is_file()}
    record("cached replay preserves baseline files exactly", baseline_after == baseline_files)
    with Path(__file__).with_name("final_study.pkl").open("wb") as stream:
        pickle.dump(final, stream)
    shutil.copy2(SOURCE, out / "validated_study.py")
    summary_path = out / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_cached_replay"] = {
        "optimizers_executed": 0, "checks_passed": 19, "accepted_endpoints": 12,
        "reflected_endpoints": reflection_count,
        "postreflection_failure_fixture_passed": True,
        "report_and_source_snapshot_refreshed": True,
    }
    for row, actual in zip(summary["endpoint_comparisons"], final["results"]):
        row["gradient_after"] = actual["reduced_gradient_inf_norm"]
        row["EL_over_h_after"] = actual["el_residual_over_h"]
    summary["checks"] = [{"name": name, "passed": bool(ok), "measured": float(value), "criterion": criterion}
                         for name, ok, value, criterion in final["checks"]]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    test_hessians_and_gauge()
    test_curvature_classification()
    test_correction()
    test_solve_integration()
    test_energy_comparison()
    test_cached_study_integration()
    output = {"python": sys.executable, "source": str(SOURCE), "count": len(RESULTS), "checks": RESULTS}
    Path(__file__).with_name("focused_validation_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"python": sys.executable, "passed": len(RESULTS), "failed": 0,
                      "max_atomistic_hessian_fd_error": max(row["measured"] for row in RESULTS if "atomistic full Hessian FD" in row["name"]),
                      "max_continuum_hessian_fd_error": max(row["measured"] for row in RESULTS if "continuum full Hessian FD" in row["name"])}))


    import shutil
    out = REPO / "outputs/relaxation/warm_start_validation_20260914_122932"
    for name in ("validate_safeguards.py", "focused_validation_results.json"):
        shutil.copy2(Path(__file__).with_name(name), out / name)

