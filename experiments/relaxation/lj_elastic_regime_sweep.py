"""Reference refinement, reporting, and bounded sweeps for the two-chain study.

The numerical model remains in atomistic_two_chain_lj_lbfgs_convergence_study.
This module keeps sweep orchestration and plots separate from its objectives.
Imports do not start computations or write files.
"""

import contextlib
import json
import multiprocessing as mp
import pickle
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.relaxation.lj_elastic_dominance import (
    atomistic_regime, continuum_regime,
)


DEFAULT_INTERACTION_SCALES = (1.0, 2.0, 3.0, 4.0, 5.0, 8.0)


def failed_atomistic_result(model, N, error):
    """Preserve an endpoint failure without substituting a successful profile."""
    scalar_keys = (
        "energy", "initial_energy", "initial_reduced_gradient_inf_norm",
        "lbfgsb_reduced_gradient_inf_norm", "raw_energy",
        "raw_reduced_gradient_inf_norm", "raw_el_residual_over_h",
        "reduced_gradient_inf_norm", "el_residual_inf_norm", "el_residual_over_h",
        "curvature_min_eigenvalue", "curvature_tolerance", "mean1", "mean2", "mean_sum",
    )
    return {
        **dict.fromkeys(scalar_keys, np.nan),
        "raw_u": np.full(2 * N + 1, np.nan),
        "normalized_u": np.full(2 * N + 1, np.nan),
        "accepted": False, "iterations": 0, "function_calls": 0, "warnflag": -1,
        "optimizer_task": str(error), "failure_reason": str(error),
        "gradient_reduction_ratio": None, "curvature_status": "unavailable",
        "newton_correction_status": "not_attempted", "newton_correction_applied": False,
        "newton_step_inf_norm": 0.0,
        "normalization": {
            "normalization_d": np.nan, "normalization_c": np.nan,
            "normalization_mean_bound": model.atomistic_geometry(N)[-1] / 4.0,
        },
    }


def _refresh_comparisons(study, model):
    """Update references and errors while keeping every atomistic endpoint fixed."""
    points = max(study["continuum_results"])
    reference = study["continuum_results"][points]
    spline = model.periodic_q_spline(reference["x"], reference["q"])
    study["reference_points"] = points
    for run in study["results"]:
        N = run["N"]
        run["continuum_pair"] = model.sample_continuum_pair(spline, N)
        run.update(model.two_layer_errors(run["normalized_u"], run["continuum_pair"], N))
        run["energy_minus_continuum"] = run["raw_energy"] - 0.5 * reference["energy"]
        matched = model.two_layer_errors(
            run["normalized_u"],
            model.sample_continuum_pair(spline, N, run["continuum_phase_shift"]), N)
        for norm in ("h2", "max"):
            run[f"phase_matched_{norm}_error"] = matched[f"{norm}_error"]
            run[f"{norm}_error_over_h"] = run[f"{norm}_error"] / run["h"]
            run[f"phase_matched_{norm}_error_over_h"] = matched[f"{norm}_error"] / run["h"]


def _reference_checks(study, model):
    """Compare the last two continuum grids for both starts, phases, and norms."""
    coarse_points, fine_points = sorted(study["continuum_results"])[-2:]
    coarse = study["continuum_results"][coarse_points]
    fine = study["continuum_results"][fine_points]
    solutions_accepted = coarse.get("accepted", True) and fine.get("accepted", True)
    coarse_spline = model.periodic_q_spline(coarse["x"], coarse["q"])
    fine_spline = model.periodic_q_spline(fine["x"], fine["q"])
    checks = []
    for run in study["results"]:
        if run["N"] != study["n_values"][-1]:
            continue
        for phase_name, shift in (("fixed", 0.0), ("phase_matched", run["continuum_phase_shift"])):
            difference = model.two_layer_errors(
                model.sample_continuum_pair(coarse_spline, run["N"], shift),
                model.sample_continuum_pair(fine_spline, run["N"], shift), run["N"])
            for norm in ("h2", "max"):
                key = f"{norm}_error" if phase_name == "fixed" else f"phase_matched_{norm}_error"
                limit = model.REFINEMENT_FRACTION * run[key]
                change = difference[f"{norm}_error"]
                checks.append({
                    "start": run["start"], "phase": phase_name, "norm": norm,
                    "coarse_points": coarse_points, "fine_points": fine_points,
                    "change": change, "limit": limit,
                    "solutions_accepted": bool(solutions_accepted),
                    "passed": bool(solutions_accepted and np.isfinite(change)
                                   and np.isfinite(limit) and change < limit),
                })
    return checks


def complete_regime_study(study, model, max_continuum_points=3200):
    """Add regime bounds and reference validation; do not rerun atomistic solves."""
    scale = study["interaction_scale"]
    study["refinement_failures"] = []
    study["continuum_failures"] = [
        {"points": points, "error": reference.get("failure_reason", "Continuum solution rejected.")}
        for points, reference in study["continuum_results"].items()
        if not reference.get("accepted", True)]
    _refresh_comparisons(study, model)
    while True:
        reference_checks = _reference_checks(study, model)
        # A rejected/nonfinite endpoint cannot drive useful reference refinement.
        accepted_starts = {r["start"] for r in study["results"]
                           if r["N"] == study["n_values"][-1] and r["accepted"]}
        final_pair_accepted = all(c["solutions_accepted"] for c in reference_checks)
        needs_refinement = (not final_pair_accepted or any(
            not c["passed"] for c in reference_checks if c["start"] in accepted_starts))
        points = 2 * study["reference_points"]
        if not needs_refinement or points > max_continuum_points:
            break
        try:
            reference = model.solve_continuum_reference(points, interaction_scale=scale)
        except model.ContinuumSolveError as error:
            failure = {"points": points, "error": str(error)}
            study["refinement_failures"].append(failure)
            study["continuum_failures"].append(failure)
            reference = error.solution
            if not (np.all(np.isfinite(reference["q"])) and np.isfinite(reference["energy"])):
                break
            # Retain a finite rejected solve for evidence, without promoting its
            # acceptance. The next ordinary grid solve still starts from zero.
            reference["accepted"] = False
            reference["failure_reason"] = str(error)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            study["refinement_failures"].append({"points": points, "error": str(error)})
            break
        try:
            _, _, curvature = model.hessian_spectrum(model.continuum_full_hessian(
                reference["q"], reference["x"], reference["dx"], interaction_scale=scale))
            reference.update(curvature)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            study["refinement_failures"].append({"points": points, "error": str(error)})
            break
        study["continuum_results"][points] = reference
        _refresh_comparisons(study, model)
    study["continuum_points"] = tuple(sorted(study["continuum_results"]))
    study["reference_checks"] = reference_checks
    study["final_continuum_pair_accepted"] = all(
        c["solutions_accepted"] for c in reference_checks)
    study["reference_resolved"] = all(c["passed"] for c in reference_checks)
    # Keep the original field, now referring to the final pair of reference grids.
    study["continuum_reference_change"] = {
        f"{norm}_error": next(c["change"] for c in reference_checks
                             if c["start"] == "sampled_continuum" and
                             c["phase"] == "fixed" and c["norm"] == norm)
        for norm in ("h2", "max")
    }

    regime = continuum_regime(model.LJ_PARAMETERS, scale, model.PAIR_CUTOFF)
    refined_bound = continuum_regime(model.LJ_PARAMETERS, scale, model.PAIR_CUTOFF, 4096)
    cutoff_bound = continuum_regime(model.LJ_PARAMETERS, scale, model.CUTOFF_CHECK)
    regime["rho_4096"] = refined_bound["rho"]
    regime["rho_cutoff_check"] = cutoff_bound["rho"]
    regime["cutoff_rho_change"] = abs(regime["rho"] - cutoff_bound["rho"])
    regime["phase_cell_covered"] = all(
        np.all(np.isfinite(c["q"])) and
        np.max(np.abs(c["x"] + c["q"])) <= np.pi + 64 * np.finfo(float).eps
        for c in study["continuum_results"].values())
    regime["established"] = regime["established"] and regime["phase_cell_covered"]
    study["continuum_regime"] = regime
    cutoff_ok = {c["start"]: c["el_difference_over_h"] <= model.CUTOFF_EL_OVER_H_TOL
                 for c in study["cutoff_evaluations"]}
    preflight_ok = all(check[1] for check in study["checks"][:5])
    continuum_cutoff_ok = regime["cutoff_rho_change"] <= 1.0e-6
    continuum_curvature_ok = all(c["curvature_status"] in ("soft", "positive")
                                 for c in study["continuum_results"].values())
    continuum_ok = (regime["phase_cell_covered"] and continuum_curvature_ok
                    and study["final_continuum_pair_accepted"])
    for run in study["results"]:
        bound = atomistic_regime(run["continuum_pair"], run["normalized_u"], run["N"],
                                 model.LJ_PARAMETERS, scale, model.PAIR_CUTOFF)
        tail = atomistic_regime(run["continuum_pair"], run["normalized_u"], run["N"],
                                model.LJ_PARAMETERS, scale, model.CUTOFF_CHECK)
        run.update({
            "interaction_scale": scale, "rho_atomistic": bound["rho"],
            "elastic_dominance_gap": bound["gap"],
            "elastic_dominance_established": bound["established"],
            "force_lipschitz_bound": bound["force_lipschitz_bound"],
            "poincare_constant": bound["poincare_constant"],
            "rho_atomistic_cutoff_check": tail["rho"],
            "regime_cutoff_change": abs(tail["rho"] - bound["rho"]),
            "reference_resolved": all(c["passed"] for c in reference_checks
                                      if c["start"] == run["start"]),
        })
        run["comparison_valid"] = bool(
            run["accepted"] and preflight_ok and continuum_ok and continuum_cutoff_ok
            and cutoff_ok[run["start"]] and run["reference_resolved"]
            and run["regime_cutoff_change"] <= 1.0e-6)

    original_checks = study["checks"]
    observation_names = {"two-start energy comparison", "warm H_h^2 monotonic decrease",
                         "warm maximum-error monotonic decrease"}
    numerical_checks = [c for c in original_checks
                        if c[0] not in observation_names and "all-grid slope" not in c[0]
                        and not (c[0].startswith("continuum ") and
                                 c[0].endswith(("H_h^2 change", "maximum change")))]
    numerical_checks.extend([
        ("continuum phase-cell coverage", regime["phase_cell_covered"],
         float(regime["phase_cell_covered"]), "== 1"),
        ("all reference comparisons resolved", study["reference_resolved"],
         float(sum(c["passed"] for c in reference_checks)), f"== {len(reference_checks)}"),
        ("final continuum pair accepted", study["final_continuum_pair_accepted"],
         float(study["final_continuum_pair_accepted"]), "== 1"),
        ("refined continuum curvature", continuum_curvature_ok,
         float(continuum_curvature_ok), "== 1"),
        ("continuum regime cutoff change", regime["cutoff_rho_change"] <= 1.0e-6,
         regime["cutoff_rho_change"], "<= 1e-6"),
        ("atomistic regime cutoff change",
         all(r["regime_cutoff_change"] <= 1.0e-6 for r in study["results"]),
         float(np.max([r["regime_cutoff_change"] for r in study["results"]])), "<= 1e-6"),
    ])
    study["numerical_checks"] = numerical_checks
    study["numerically_valid"] = all(c[1] for c in numerical_checks)
    # Preserve the original baseline expectations as explicitly separate checks.
    study["baseline_checks"] = original_checks if scale == 1.0 else []
    study["checks"] = numerical_checks
    study["finest_slopes"] = {}
    study["valid_subset_slopes"] = {}
    study["fit_n_values"] = {}
    for start in model.ATOMISTIC_START_NAMES:
        runs = [r for r in study["results"] if r["start"] == start]
        valid_runs = [r for r in runs if r["comparison_valid"] and all(
            np.isfinite(r[f"{key}_error"]) and r[f"{key}_error"] > 0
            for key in ("h2", "max", "phase_matched_h2", "phase_matched_max"))]
        subset = valid_runs if len(valid_runs) >= 3 else []
        study["fit_n_values"][start] = [r["N"] for r in subset]
        for target, selected in ((study["slopes"], runs),
                                 (study["finest_slopes"], runs[-3:]),
                                 (study["valid_subset_slopes"], subset)):
            target[start] = {}
            for key in ("h2", "max", "phase_matched_h2", "phase_matched_max"):
                errors = [r[f"{key}_error"] for r in selected]
                valid = (len(selected) >= 3 and all(r["comparison_valid"] for r in selected)
                         and np.all(np.isfinite(errors)) and np.all(np.asarray(errors) > 0))
                target[start][key] = (model.loglog_slope([r["N"] for r in selected], errors)
                                      if valid else np.nan)
    # These observations use the final reference and valid fits; only the
    # explicitly retained baseline checks describe the original comparisons.
    observations = [c for c in original_checks if c[0] == "two-start energy comparison"]
    for start, slopes in study["slopes"].items():
        for norm in ("h2", "max"):
            value = slopes[norm]
            observations.append((
                f"{start} {norm} all-grid slope",
                bool(model.SLOPE_INTERVAL[0] <= value <= model.SLOPE_INTERVAL[1]), value,
                f"in [{model.SLOPE_INTERVAL[0]:.1f}, {model.SLOPE_INTERVAL[1]:.1f}]; "
                "unavailable unless every grid is valid"))
    warm = [r for r in study["results"] if r["start"] == "sampled_continuum"]
    for norm, label in (("h2", "H_h^2"), ("max", "maximum-error")):
        differences = np.diff([r[f"{norm}_error"] for r in warm])
        available = (differences.size > 0 and np.all(np.isfinite(differences))
                     and all(r["comparison_valid"] for r in warm))
        observations.append((
            f"warm {label} monotonic decrease",
            bool(available and np.all(differences < 0)),
            float(np.max(differences)) if available else np.nan,
            "< 0; unavailable unless every grid is valid"))
    study["observational_checks"] = observations
    return study


def _json_value(value):
    """Convert numerical report metadata to strict JSON."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    """Write atomically, briefly retrying Windows access/sharing conflicts."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            # A Windows reader can temporarily block replacement of a closed file.
            # Keep both payloads intact and propagate persistent or unrelated errors.
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


def _case_worker(scale, n_values, continuum_points, max_points, outdir):
    """Run one independent case in a process that the sweep can time-limit."""
    from experiments.relaxation import atomistic_two_chain_lj_lbfgs_convergence_study as model

    outdir = Path(outdir)
    started = time.monotonic()
    with (outdir / "run.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                study = model.run_study(scale, n_values, continuum_points, max_points)
                temporary = outdir / "study.pkl.tmp"
                with temporary.open("wb") as stream:
                    pickle.dump(study, stream)
                temporary.replace(outdir / "study.pkl")
                model.save_summary(outdir, study)
                model.save_profiles(outdir, study)
                model.save_plots(outdir, study)
                model.save_report(outdir, study, study["numerically_valid"])
                metadata = {
                    "status": "complete", "interaction_scale": scale,
                    "numerically_valid": study["numerically_valid"],
                    "baseline_checks_passed": all(c[1] for c in study["baseline_checks"])
                                              if study["baseline_checks"] else None,
                    "continuum_regime": study["continuum_regime"],
                    "reference_points": study["reference_points"],
                    "warm_initial_continuum_points": study["warm_initial_continuum_points"],
                    "reference_checks": study["reference_checks"],
                    "refinement_failures": study["refinement_failures"],
                    "continuum_failures": study["continuum_failures"],
                    "final_continuum_pair_accepted": study["final_continuum_pair_accepted"],
                    "numerical_checks": study["numerical_checks"],
                    "observational_checks": study["observational_checks"],
                    "slopes": study["slopes"], "finest_slopes": study["finest_slopes"],
                    "valid_subset_slopes": study["valid_subset_slopes"],
                    "fit_n_values": study["fit_n_values"],
                }
            except Exception as error:
                traceback.print_exc()
                metadata = {"status": "failed", "interaction_scale": scale,
                            "error": f"{type(error).__name__}: {error}"}
            metadata["warnings"] = [str(w.message) for w in captured]
            metadata["elapsed_seconds"] = time.monotonic() - started
            for message in metadata["warnings"]:
                print("WARNING:", message)
            write_json(outdir / "case_status.json", metadata)


def save_sweep_outputs(outdir, cases, model):
    """Save cross-parameter tables/figures without promoting failed endpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    studies = []
    for case in cases:
        if case["status"] == "complete":
            with (outdir / case["directory"] / "study.pkl").open("rb") as stream:
                studies.append(pickle.load(stream))
    unfinished = [f"{case['interaction_scale']:g}" for case in cases if case["status"] != "complete"]
    missing_note = "; unfinished lambda=" + ", ".join(unfinished) if unfinished else ""
    by_scale = {study["interaction_scale"]: study for study in studies}
    case_rows = []
    for case in cases:
        scale = case["interaction_scale"]
        study = by_scale.get(scale)
        bound = continuum_regime(model.LJ_PARAMETERS, scale, model.PAIR_CUTOFF)
        row = {
            "interaction_scale": scale, "status": case["status"],
            "numerically_valid": case.get("numerically_valid", False),
            "rho_continuum_cell": bound["rho"],
            "continuum_cell_gap": bound["gap"],
            "reference_points": study["reference_points"] if study else None,
            "accepted_endpoints": sum(r["accepted"] for r in study["results"]) if study else None,
            "completed_endpoints": len(study["results"]) if study else 0,
            "error": case.get("error", ""),
        }
        for start, tag in zip(model.ATOMISTIC_START_NAMES, ("warm", "seed3")):
            values = [r["rho_atomistic"] for r in study["results"] if r["start"] == start] if study else []
            row[f"max_rho_{tag}"] = float(np.max(values)) if values else np.nan
        case_rows.append(row)
    model.write_csv(outdir / "case_summary.csv", case_rows)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    scales = [row["interaction_scale"] for row in case_rows]
    ax.plot(scales, [row["rho_continuum_cell"] for row in case_rows], "o-", label="Continuum phase-cell bound")
    for tag, label in (("warm", "Warm: maximum over N"), ("seed3", "Seed 3: maximum over N")):
        ax.plot(scales, [row[f"max_rho_{tag}"] for row in case_rows], "s--", label=label)
    ax.axhline(1, color="black", linestyle=":", label="Sufficient-condition threshold")
    ax.set(xlabel=r"Interaction multiplier $\lambda$", ylabel="Conservative dominance bound",
           title="A bound above one does not establish instability")
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "regime_bounds.png", dpi=180)
    plt.close(fig)
    if not studies:
        return
    rows, slopes = [], []
    for study in studies:
        for run in study["results"]:
            rows.append({"interaction_scale": study["interaction_scale"],
                         "rho_continuum": study["continuum_regime"]["rho"],
                         **{key: run[key] for key in (
                             "N", "start", "accepted", "comparison_valid", "rho_atomistic",
                             "elastic_dominance_gap", "elastic_dominance_established",
                             "reference_resolved", "h2_error_over_h", "max_error_over_h",
                             "phase_matched_h2_error_over_h", "phase_matched_max_error_over_h")}})
        for start in model.ATOMISTIC_START_NAMES:
            slopes.append({"interaction_scale": study["interaction_scale"], "start": start,
                           **{f"all_{key}": value for key, value in study["slopes"][start].items()},
                           **{f"finest3_{key}": value for key, value in study["finest_slopes"][start].items()},
                           **{f"valid_subset_{key}": value for key, value in study["valid_subset_slopes"][start].items()},
                           "valid_subset_n_values": " ".join(map(str, study["fit_n_values"][start]))})
    model.write_csv(outdir / "regime_summary.csv", rows)
    model.write_csv(outdir / "slope_summary.csv", slopes)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(.08, .88, len(studies)))
    for study, color in zip(studies, colors):
        for column, start in enumerate(model.ATOMISTIC_START_NAMES):
            runs = [r for r in study["results"] if r["start"] == start]
            for row, norm in enumerate(("h2", "max")):
                for prefix, style in (("", "--"), ("phase_matched_", "-")):
                    errors = [r[f"{prefix}{norm}_error"] if r["comparison_valid"] else np.nan for r in runs]
                    axes[row, column].loglog([r["N"] for r in runs], errors, style,
                                            color=color, marker="o", markersize=3,
                                            label=f"lambda={study['interaction_scale']:g}" if prefix else None)
                invalid = [r for r in runs if not r["comparison_valid"] and
                           np.isfinite(r[f"phase_matched_{norm}_error"])]
                axes[row, column].scatter([r["N"] for r in invalid],
                                         [r[f"phase_matched_{norm}_error"] for r in invalid],
                                         marker="x", color=color, alpha=.5)
    for column, title in enumerate(("Continuum warm start", "Seed-3 start")):
        axes[0, column].set_title(title)
        axes[1, column].set_xlabel("Layer-1 atoms N")
        for row, label in enumerate((r"$H_h^2$ error", "Maximum error")):
            axes[row, column].set_ylabel(label)
            axes[row, column].grid(True, which="both", alpha=.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.legend(handles=[Line2D([0], [0], color="black", label="Phase matched"),
                        Line2D([0], [0], color="black", linestyle="--", label="Fixed phase"),
                        Line2D([0], [0], color="gray", marker="x", linestyle="", label="Unresolved / rejected")],
               loc="lower center", ncol=3)
    fig.suptitle("Atomistic-to-continuum convergence across interaction strengths" + missing_note)
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(outdir / "convergence_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(studies), figsize=(3.3 * len(studies), 7),
                             squeeze=False, sharex=True, sharey="row")
    for column, study in enumerate(studies):
        N = study["n_values"][-1]
        continuum = study["continuum_results"][study["reference_points"]]
        spline = model.periodic_q_spline(continuum["x"], continuum["q"])
        for run, color in zip([r for r in study["results"] if r["N"] == N], ("tab:blue", "tab:orange")):
            label = "warm" if run["start"] == "sampled_continuum" else "seed 3"
            if not run["comparison_valid"]:
                label += " (unresolved)"
            sampled = model.sample_continuum_pair(spline, N, run["continuum_phase_shift"])
            for row, (profile, reference) in enumerate(zip(model.split_layers(run["normalized_u"], N),
                                                          model.split_layers(sampled, N))):
                x = np.arange(profile.size) * model.TWOPI / profile.size
                axes[row, column].plot(x, profile, color=color, label=label)
                axes[row, column].plot(x, reference, "--", color=color, alpha=.8, label=label + " continuum")
        unresolved = any(not r["comparison_valid"] for r in study["results"] if r["N"] == N)
        title = f"lambda={study['interaction_scale']:g}, N={N}"
        if unresolved:
            title += "\nUnresolved comparison"
        axes[0, column].set_title(title, fontsize=10, color="tab:red" if unresolved else "black")
        for row in (0, 1):
            axes[row, column].set(xlabel="x", ylabel=f"Layer {row + 1} displacement")
            axes[row, column].grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Atomistic profiles and phase-matched continuum references" + missing_note)
    fig.tight_layout()
    fig.savefig(outdir / "profile_comparison.png", dpi=180)
    plt.close(fig)


def run_sweep(model, interaction_scales=DEFAULT_INTERACTION_SCALES, outdir=None,
              n_values=None, continuum_points=None, max_continuum_points=3200,
              time_budget_seconds=900):
    """Run independent scales with a hard wall-clock budget for the calculations.

Completed cases are checkpointed. A timed-out worker is terminated; table and
plot generation from completed checkpoints takes place after the compute budget.
"""
    scales = tuple(float(s) for s in interaction_scales)
    if not scales or len(set(scales)) != len(scales) or any(not np.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("Interaction scales must be distinct, finite, and positive.")
    if not np.isfinite(time_budget_seconds) or time_budget_seconds <= 0:
        raise ValueError("The compute time budget must be finite and positive.")
    n_values = tuple(model.N_VALUES if n_values is None else n_values)
    continuum_points = tuple(model.CONTINUUM_POINTS if continuum_points is None else continuum_points)
    if outdir is None:
        outdir = (model.REPOSITORY_ROOT / "outputs" / "relaxation" /
                  "output_atomistic_two_chain_lj_lbfgs_convergence_study" /
                  ("elastic_regime_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    cases = [{"interaction_scale": scale, "status": "pending", "directory": f"scale_{scale:.17g}"}
             for scale in scales]
    metadata = {"interaction_scales": scales, "n_values": n_values,
                "initial_continuum_points": continuum_points, "max_continuum_points": max_continuum_points,
                "time_budget_seconds": time_budget_seconds, "cases": cases}
    started = time.monotonic()
    for case in cases:
        remaining = time_budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            case["status"] = "not_run_budget_exhausted"
            continue
        directory = outdir / case["directory"]
        directory.mkdir()
        case["status"] = "running"
        write_json(outdir / "sweep_status.json", metadata)
        print(f"lambda={case['interaction_scale']:g}: starting ({remaining:.0f}s remain)", flush=True)
        process = mp.get_context("spawn").Process(target=_case_worker, args=(
            case["interaction_scale"], n_values, continuum_points, max_continuum_points, str(directory)))
        process.start()
        process.join(timeout=max(0, time_budget_seconds - (time.monotonic() - started)))
        if process.is_alive():
            process.terminate()
            process.join()
            case["status"] = "timed_out"
            write_json(directory / "case_status.json", case)
        elif (directory / "case_status.json").exists():
            case.update(json.loads((directory / "case_status.json").read_text(encoding="utf-8")))
        else:
            case.update(status="failed", error=f"Worker exited with code {process.exitcode}")
        write_json(outdir / "sweep_status.json", metadata)
        print(f"lambda={case['interaction_scale']:g}: {case['status']}; "
              f"numerically_valid={case.get('numerically_valid', 'unavailable')}", flush=True)
    metadata["compute_elapsed_seconds"] = time.monotonic() - started
    write_json(outdir / "sweep_status.json", metadata)
    save_sweep_outputs(outdir, cases, model)
    return {"outdir": str(outdir), **metadata}
