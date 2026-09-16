"""Validate the phase-reference change using cached endpoints and five N=20 solves."""
import contextlib
import copy
import csv
import difflib
import hashlib
import io
import json
import pickle
import re
import time
import types
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / "experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py"
CANONICAL = ROOT / "outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study"
CACHE = Path(r"C:\Users\lh201\AppData\Local\Temp\clr_warm_start_20260914_122932\final_study.pkl")
BEFORE = (OUT / "before_study.py").read_text(encoding="utf-8")
AFTER = SOURCE.read_text(encoding="utf-8")
NEW_FIELDS = ("continuum_phase_shift", "phase_matched_h2_error", "phase_matched_max_error")


def load_module(source, name, seed=None):
    if seed is not None:
        source, count = re.subn(r"^ATOMISTIC_RANDOM_FOURIER_SEED = 3$",
                               f"ATOMISTIC_RANDOM_FOURIER_SEED = {seed}", source,
                               count=1, flags=re.MULTILINE)
        assert count == 1
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    return module


with CACHE.open("rb") as stream:
    CACHED = pickle.load(stream)
with np.load(CANONICAL / "profiles.npz") as archive:
    ARRAYS = {key: archive[key].copy() for key in archive.files}
for result in CACHED["results"]:
    N, start = result["N"], result["start"]
    np.testing.assert_array_equal(result["raw_u"], ARRAYS[f"raw_u_atomistic_{start}_N{N}"])
for points, continuum in CACHED["continuum_results"].items():
    np.testing.assert_array_equal(continuum["q"], ARRAYS[f"continuum_q_N{points}"])


def replay(module, directory, *, new=False, seed=3, reject_first=False):
    """Replay the production aggregation; optimizers are forbidden."""
    directory.mkdir(exist_ok=True)
    endpoints = copy.deepcopy(CACHED["results"])
    for index, result in enumerate(endpoints):
        result["normalized_u"], result["normalization"] = module.appendix_b_normalize(result["raw_u"], result["N"])
        if result["start"] != "sampled_continuum":
            result["start"] = f"random_fourier_seed_{seed}"
        if reject_first and index == 0:
            result["accepted"] = False
    iterator = iter(endpoints)

    def endpoint(initial, N, cutoff):
        result = next(iterator)
        assert N == result["N"] and cutoff == module.PAIR_CUTOFF
        return result

    with patch.object(module, "solve_continuum_reference", side_effect=lambda points: copy.deepcopy(CACHED["continuum_results"][points])) as continuum_solver, \
            patch.object(module, "solve_atomistic_initial", side_effect=endpoint) as atomistic_solver, \
            patch.object(module.opt, "fmin_l_bfgs_b", side_effect=AssertionError("Optimizer forbidden in replay")) as optimizer:
        study = module.run_study()
        assert optimizer.call_count == 0
        assert continuum_solver.call_count == 2 and atomistic_solver.call_count == 12
    module.save_summary(directory, study)
    module.save_profiles(directory, study)
    with contextlib.redirect_stdout(io.StringIO()):
        module.save_report(directory, study, all(item[1] for item in study["checks"]))
    if new:
        assert all(all(key in item for key in NEW_FIELDS) for item in study["results"])
    return study


def read_csv(directory):
    with (directory / "convergence_summary.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


started = time.perf_counter()
summary = {"replay_optimizer_calls": 0}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    before = load_module(BEFORE, "before")
    after = load_module(AFTER, "after")
    continuum = CACHED["continuum_results"][800]
    spline = after.periodic_q_spline(continuum["x"], continuum["q"])

    # Independent analytic periodic profile, including shifts crossing both cell boundaries.
    analytic_cases = []
    for N in (20, 21):
        x1 = 2 * np.pi * np.arange(N) / N
        x2 = 2 * np.pi * np.arange(N + 1) / (N + 1)
        def q(x):
            assert np.all(x >= -np.pi) and np.all(x < np.pi)
            return 0.7 * np.sin(x) + 0.2 * np.sin(2 * x)
        for delta in (0.0, 0.137, -0.219, 2 * np.pi + 0.137, -2 * np.pi - 0.219):
            actual = after.sample_continuum_pair(q, N, delta)
            exact1 = (0.7 * np.sin(x1 - delta) + 0.2 * np.sin(2 * (x1 - delta)) - delta) / 2
            exact2 = -(0.7 * np.sin(x2 - delta) + 0.2 * np.sin(2 * (x2 - delta)) - delta) / 2
            expected = np.concatenate((exact1, exact2))
            # Trigonometric argument reduction only: a few ulps at O(1) scale.
            np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-15)
            np.testing.assert_allclose([np.mean(actual[:N]), np.mean(actual[N:])],
                                       [-delta / 2, delta / 2], rtol=0, atol=4e-15)
            analytic_cases.append({"N": N, "phase": delta, "max_error": float(np.max(np.abs(actual - expected)))})
    for N in after.N_VALUES:
        np.testing.assert_array_equal(before.sample_continuum_pair(spline, N), after.sample_continuum_pair(spline, N))
        np.testing.assert_array_equal(after.sample_continuum_pair(spline, N), after.sample_continuum_pair(spline, N, 0.0))
    summary["analytic_phase_cases"] = analytic_cases
    summary["zero_shift_bitwise_compatible"] = True

    baseline = replay(before, OUT / "baseline")
    updated = replay(after, OUT, new=True)
    assert baseline["checks"] == updated["checks"]
    assert baseline["energy_comparisons"] == updated["energy_comparisons"]
    for start, slopes in baseline["slopes"].items():
        assert all(updated["slopes"][start][key] == value for key, value in slopes.items())
        assert set(updated["slopes"][start]) == set(slopes) | {"phase_matched_h2", "phase_matched_max"}
    old_csv, new_csv = read_csv(OUT / "baseline"), read_csv(OUT)
    assert len(new_csv) == 12
    assert list(new_csv[0]) == list(old_csv[0]) + list(NEW_FIELDS)
    for old, new in zip(old_csv, new_csv):
        assert {key: new[key] for key in old} == old
    errors = []
    for old, new in zip(baseline["results"], updated["results"]):
        assert set(new) == set(old) | set(NEW_FIELDS)
        for key, value in old.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, new[key])
            else:
                assert value == new[key], key
        N = new["N"]
        u1, u2 = new["normalized_u"][:N], new["normalized_u"][N:]
        delta = np.mean(u2) - np.mean(u1)
        assert new["continuum_phase_shift"] == delta
        # Independent reconstruction from the periodic spline, without the changed sampler.
        x1, x2 = 2 * np.pi * np.arange(N) / N, 2 * np.pi * np.arange(N + 1) / (N + 1)
        r1 = (spline((x1 - delta + np.pi) % (2 * np.pi) - np.pi) - delta) / 2
        r2 = -(spline((x2 - delta + np.pi) % (2 * np.pi) - np.pi) - delta) / 2
        reference = np.concatenate((r1, r2))
        expected = before.two_layer_errors(new["normalized_u"], reference, N)
        np.testing.assert_allclose([new["phase_matched_h2_error"], new["phase_matched_max_error"]],
                                   [expected["h2_error"], expected["max_error"]], rtol=2e-10, atol=2e-13)
        if new["start"] == after.ATOMISTIC_RANDOM_START_NAME:
            errors.append({"N": N, **{key: new[key] for key in NEW_FIELDS}})
    np.testing.assert_allclose([errors[0]["phase_matched_h2_error"], errors[-1]["phase_matched_h2_error"]],
                               [0.02038634, 0.00064299254], rtol=0, atol=5e-9)
    with np.load(OUT / "baseline/profiles.npz") as old, np.load(OUT / "profiles.npz") as new:
        assert old.files == new.files == list(ARRAYS)
        for key in old.files:
            np.testing.assert_array_equal(old[key], new[key])
            np.testing.assert_array_equal(new[key], ARRAYS[key])
    old_report = (OUT / "baseline/check_report.txt").read_text(encoding="utf-8")
    report = (OUT / "check_report.txt").read_text(encoding="utf-8")
    # The original six-row profile table and energy table remain byte-for-byte in the new report.
    def section(text, title, end):
        return text[text.index(title):text.index(end, text.index(title))]
    profile_table = section(old_report, "Errors and seed-3 normalized mean", "\nEnergy differences to continuum").rstrip()
    assert profile_table in report
    assert section(old_report, "Energy differences to continuum", "\nAll-grid log(error)") in report
    for item in errors:
        tokens = [f"{item['continuum_phase_shift']:+.6e}", f"{item['phase_matched_h2_error']:.4e}", f"{item['phase_matched_max_error']:.4e}"]
        matches = [line for line in report.splitlines() if line.split() == [str(item["N"])] + tokens]
        assert len(matches) == 1, (item["N"], tokens)
    with patch.object(after.opt, "fmin_l_bfgs_b", side_effect=AssertionError("Optimizer forbidden while saving plots")):
        after.save_plots(OUT, updated)
    summary["seed3_phase_matched_errors"] = errors
    summary["phase_matched_slopes"] = {key: value for key, value in updated["slopes"][after.ATOMISTIC_RANDOM_START_NAME].items() if key.startswith("phase_matched")}
    summary["existing_values_arrays_checks_and_default_csv_fields_preserved"] = True
    summary["old_profile_and_energy_report_tables_preserved"] = True

    # Check the existing slope-acceptance rule for the new fields, without solving.
    failed = replay(after, OUT / "unaccepted_fixture", new=True, reject_first=True)
    assert all(np.isnan(failed["slopes"]["sampled_continuum"][key])
               for key in ("h2", "max", "phase_matched_h2", "phase_matched_max"))
    summary["unaccepted_track_slopes_unavailable"] = True

    # Exercise actual initializations/endpoints once per seed; no restarts or larger solves.
    seed_cases = []
    solve_started = time.perf_counter()
    for seed in range(5):
        assert time.perf_counter() - solve_started < 60, "Five-seed validation budget exhausted"
        module = load_module(AFTER, f"seed_{seed}", seed)
        assert module.ATOMISTIC_RANDOM_START_NAME == f"random_fourier_seed_{seed}"
        assert module.ATOMISTIC_RANDOM_SEED_TAG == f"seed{seed}"
        reference = module.sample_continuum_pair(spline, 20)
        initials = module.atomistic_initial_guesses(20, reference)
        assert tuple(initials) == ("sampled_continuum", f"random_fourier_seed_{seed}")
        initial = initials[module.ATOMISTIC_RANDOM_START_NAME]
        began = time.perf_counter()
        result = module.solve_atomistic_initial(initial, 20, module.PAIR_CUTOFF)
        u, _, reflected = module.reflection_aligned_errors(result["normalized_u"], reference, 20)
        snapshot = u.copy()
        delta = float(np.mean(u[20:]) - np.mean(u[:20]))
        matched = module.sample_continuum_pair(spline, 20, delta)
        metrics = module.two_layer_errors(u, matched, 20)
        np.testing.assert_array_equal(u, snapshot)
        assert np.isfinite(delta) and all(np.isfinite(value) for value in metrics.values())
        seed_cases.append({"seed": seed, "seconds": time.perf_counter() - began, "accepted": bool(result["accepted"]),
                           "correction": result["newton_correction_status"], "phase": delta, "reflection_applied": reflected,
                           "gradient": result["reduced_gradient_inf_norm"], "el_over_h": result["el_residual_over_h"], **metrics})
    assert time.perf_counter() - solve_started < 60
    summary["actual_seed_solves"] = seed_cases
    summary["actual_seed_solve_calls"] = 5
    summary["five_seed_seconds"] = time.perf_counter() - solve_started

    # Generic naming is tested separately using cached profiles, not extra optimizations.
    for seed in (0, 1, 2, 4):
        module = load_module(AFTER, f"names_{seed}", seed)
        directory = OUT / f"seed{seed}_naming_fixture"
        study = replay(module, directory, new=True, seed=seed)
        tag, start = f"seed{seed}", f"random_fourier_seed_{seed}"
        assert set(study["slopes"]) == {"sampled_continuum", start}
        rows = read_csv(directory)
        assert f"{tag}_minus_warm_energy" in rows[0]
        assert "seed3_minus_warm_energy" not in rows[0]
        assert {row["start"] for row in rows} == {"sampled_continuum", start}
        report_text = (directory / "check_report.txt").read_text(encoding="utf-8")
        assert not re.search(r"seed[-_ ]?3|Seed-3", report_text)
        assert start in report_text and f"{tag}_H2" in report_text
        with np.load(directory / "profiles.npz") as archive:
            assert int(archive["random_fourier_seed"]) == seed
            assert all("random_fourier_seed_3" not in key for key in archive.files)
            assert any(start in key for key in archive.files)
        # A feasible lower-energy result must receive the configured status name.
        warm = copy.deepcopy(study["results"][0])
        random = copy.deepcopy(study["results"][1])
        random["raw_energy"] = warm["raw_energy"] - 1
        comparison = module.compare_atomistic_energies(warm, random, 20)
        assert comparison["energy_comparison_status"] == f"{tag}_lower"
    summary["configurable_seed_naming_0_through_4"] = True

assert not caught, [str(item.message) for item in caught]
hashes = json.loads((OUT / "preserved_file_hashes.json").read_text(encoding="utf-8"))
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
summary["existing_output_files_unchanged"] = True
summary["warnings"] = []
summary["total_seconds"] = time.perf_counter() - started
(OUT / "validated_study.py").write_bytes(SOURCE.read_bytes())
(OUT / "task.diff").write_text("".join(difflib.unified_diff(BEFORE.splitlines(True), AFTER.splitlines(True), fromfile="before", tofile="after")), encoding="utf-8")
(OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
