"""Replay saved endpoints through aggregation and output without optimization."""
import ast
import contextlib
import copy
import csv
import difflib
import hashlib
import io
import json
import pickle
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
before_text = (OUT / "before_study.py").read_text(encoding="utf-8")
after_text = SOURCE.read_text(encoding="utf-8")
before, after = types.ModuleType("before"), types.ModuleType("after")
for module, source in ((before, before_text), (after, after_text)):
    module.__file__ = str(SOURCE)
    exec(compile(source, str(SOURCE), "exec"), module.__dict__)
with CACHE.open("rb") as stream:
    cached = pickle.load(stream)
with np.load(CANONICAL / "profiles.npz") as archive:
    arrays = {key: archive[key].copy() for key in archive.files}

# Establish that the cache still represents the current saved profiles.
for result in cached["results"]:
    N, start = result["N"], result["start"]
    np.testing.assert_array_equal(result["raw_u"], arrays[f"raw_u_atomistic_{start}_N{N}"])
for points, continuum in cached["continuum_results"].items():
    np.testing.assert_array_equal(continuum["q"], arrays[f"continuum_q_N{points}"])
    np.testing.assert_array_equal(continuum["x"], arrays[f"continuum_x_N{points}"])

# Direct two-layer finite differences and Phi=2*sum(V), independently of W.
conversion_errors = {}
direct_energies = {}
for points in after.CONTINUUM_POINTS:
    x, q = arrays[f"continuum_x_N{points}"], arrays[f"continuum_q_N{points}"]
    dx = after.TWOPI / points
    u1, u2 = q / 2, -q / 2
    d1, d2 = (np.roll(u1, -1) - u1) / dx, (np.roll(u2, -1) - u2) / dx
    arguments = (x + u1 - u2)[:, None] - after.TWOPI * np.arange(-after.PAIR_CUTOFF, after.PAIR_CUTOFF + 1)
    phi = 2 * np.sum(after.V(arguments), axis=1)
    direct = float(dx * np.sum(0.5 * d1**2 + 0.5 * d2**2 + phi))
    half = 0.5 * cached["continuum_results"][points]["energy"]
    tolerance = 64 * np.finfo(float).eps * max(1, abs(direct), abs(half))
    assert abs(direct - half) <= tolerance
    conversion_errors[points] = abs(direct - half)
    direct_energies[points] = direct

replays = []
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for module, directory in ((before, OUT / "baseline"), (after, OUT)):
        directory.mkdir(exist_ok=True)
        endpoints = copy.deepcopy(cached["results"])
        for result in endpoints:
            result["normalized_u"], result["normalization"] = module.appendix_b_normalize(result["raw_u"], result["N"])
        iterator = iter(endpoints)

        def replay_endpoint(initial, N, cutoff):
            result = next(iterator)
            assert N == result["N"] and cutoff == module.PAIR_CUTOFF
            return result

        with patch.object(module, "solve_continuum_reference", side_effect=lambda points: copy.deepcopy(cached["continuum_results"][points])) as continuum_solver, \
                patch.object(module, "solve_atomistic_initial", side_effect=replay_endpoint) as atomistic_solver, \
                patch.object(module.opt, "fmin_l_bfgs_b", side_effect=AssertionError("Optimization forbidden in replay")) as optimizer:
            study = module.run_study()
            assert optimizer.call_count == 0
            assert continuum_solver.call_count == 2 and atomistic_solver.call_count == 12
        module.save_summary(directory, study)
        module.save_profiles(directory, study)
        with contextlib.redirect_stdout(io.StringIO()):
            module.save_report(directory, study, all(check[1] for check in study["checks"]))
        replays.append(study)
assert not caught, [str(w.message) for w in caught]

baseline, updated = replays
assert baseline["checks"] == updated["checks"]
assert baseline["slopes"] == updated["slopes"]
assert baseline["energy_comparisons"] == updated["energy_comparisons"]
csv_rows = []
for directory in (OUT / "baseline", OUT):
    with (directory / "convergence_summary.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows.append(list(csv.DictReader(stream)))
old_rows, new_rows = csv_rows
assert len(new_rows) == 12
assert list(new_rows[0]) == list(old_rows[0]) + ["energy_minus_continuum"]
differences = []
for old, new, result in zip(old_rows, new_rows, updated["results"]):
    assert {key: new[key] for key in old} == old
    raw_energy, _ = after.raw_atomistic_value_gradient(result["raw_u"], result["N"], after.PAIR_CUTOFF)
    assert raw_energy == result["raw_energy"]
    expected = raw_energy - direct_energies[800]
    assert abs(result["energy_minus_continuum"] - expected) <= 64 * np.finfo(float).eps * max(1, abs(raw_energy))
    assert float(new["energy_minus_continuum"]) == result["energy_minus_continuum"]
    differences.append({"N": result["N"], "start": result["start"], "energy_minus_continuum": result["energy_minus_continuum"]})

with np.load(OUT / "baseline/profiles.npz") as old, np.load(OUT / "profiles.npz") as new:
    assert old.files == new.files == list(arrays)
    for key in old.files:
        np.testing.assert_array_equal(old[key], new[key])
        np.testing.assert_array_equal(new[key], arrays[key])

old_report = (OUT / "baseline/check_report.txt").read_text(encoding="utf-8")
report = (OUT / "check_report.txt").read_text(encoding="utf-8")
start = report.index("\nEnergy differences to continuum")
end = report.index("\nAll-grid log(error)", start)
assert report[:start] + report[end:] == old_report
block = report[start:end].strip().splitlines()
assert len(block) == 9 and "800 points" in block[1] and "F[q]/2" in block[1]
for N, line in zip(after.N_VALUES, block[3:]):
    tokens = line.split()
    assert int(tokens[0]) == N
    pair = [r for r in updated["results"] if r["N"] == N]
    assert [r["start"] for r in pair] == list(after.ATOMISTIC_START_NAMES)
    assert tokens[1:] == [f"{r['energy_minus_continuum']:+.6e}" for r in pair]

# Ensure the only production edits are inside the three approved functions.
trees = [ast.parse(text) for text in (before_text, after_text)]
for tree in trees:
    tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name in {"run_study", "save_summary", "save_report"})]
assert ast.dump(trees[0]) == ast.dump(trees[1])
hashes = json.loads((OUT / "preserved_file_hashes.json").read_text())
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
(OUT / "task.diff").write_text("".join(difflib.unified_diff(before_text.splitlines(True), after_text.splitlines(True), fromfile="before", tofile="after")), encoding="utf-8")
(OUT / "validated_study.py").write_bytes(SOURCE.read_bytes())
summary = {
    "optimizer_calls": 0,
    "continuum_two_layer_energy": direct_energies[800],
    "half_energy_conversion_absolute_errors": conversion_errors,
    "energy_differences_checked": len(differences),
    "report_rows_checked": 6,
    "existing_csv_values_unchanged": True,
    "existing_report_sections_unchanged": True,
    "npz_keys_and_arrays_unchanged": True,
    "checks_and_slopes_unchanged": True,
    "existing_output_files_unchanged": True,
    "warnings": [],
    "differences": differences,
}
(OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
