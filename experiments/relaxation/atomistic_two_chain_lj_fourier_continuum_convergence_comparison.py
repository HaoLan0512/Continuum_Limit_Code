"""Compare saved atomistic LJ minimizers with an analytic Fourier continuum.

Method:
    Reuse the validated atomistic profiles, solve the phase-fixed continuum
    problem with the Poisson-summed LJ potential, and recompute the paper's
    discrete error norms and convergence rates.
Purpose:
    Confirm that a five-mode analytic Fourier evaluation gives the same
    atomistic-to-continuum comparison as the real-image continuum potential.
Inputs:
    warm-track aliases in profiles.npz from
    atomistic_two_chain_lj_lbfgs_convergence_study.py;
    a=1, sigma=0.9, L=1, epsilon=0.5; Fourier cutoffs K=5 and K=6.
Outputs:
    A convergence CSV, NPZ profiles, PNG figure, and verification report.
Output location:
    outputs/relaxation/
    output_atomistic_two_chain_lj_fourier_continuum_convergence_comparison.
Dependencies:
    clr.potentials.lj_periodic and the established atomistic study helpers.
Related files:
    atomistic_two_chain_lj_lbfgs_convergence_study.py produces the input
    atomistic minimizers and retains the M=80 real-image continuum default.
"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

from clr.potentials.lj_periodic import (
    fourier_coefficients,
    fourier_W,
    fourier_Wprime,
    periodized_W,
    periodized_Wprime,
)
from experiments.relaxation.atomistic_two_chain_lj_lbfgs_convergence_study import (
    CONTINUUM_POINTS,
    LJ_PARAMETERS,
    N_VALUES,
    atomistic_geometry,
    loglog_slope,
    periodic_q_spline,
    sample_continuum_pair,
    solve_continuum_reference,
    split_layers,
    two_layer_errors,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FOURIER_MODES = 5
REFERENCE_FOURIER_MODES = 6
REAL_IMAGE_CUTOFF = 160
REFINEMENT_FRACTION = 0.05
FOURIER_GRADIENT_TOL = 3.0e-7
FOURIER_EL_RESIDUAL_TOL = 2.0e-5

FOURIER_CONSTANT, COEFFICIENTS_6 = fourier_coefficients(
    LJ_PARAMETERS, REFERENCE_FOURIER_MODES)
COEFFICIENTS_5 = COEFFICIENTS_6[:FOURIER_MODES]


def W5(s):
    """Return the five-mode pair-derived continuum potential."""
    return fourier_W(s, FOURIER_CONSTANT, COEFFICIENTS_5)


def Wprime5(s):
    """Return the derivative of the five-mode continuum potential."""
    return fourier_Wprime(s, COEFFICIENTS_5)


def W6(s):
    """Return the six-mode reference Fourier potential."""
    return fourier_W(s, FOURIER_CONSTANT, COEFFICIENTS_6)


def Wprime6(s):
    """Return the derivative of the six-mode reference potential."""
    return fourier_Wprime(s, COEFFICIENTS_6)


def W_image(s):
    """Return the M=160 real-image reference potential."""
    return periodized_W(s, LJ_PARAMETERS, REAL_IMAGE_CUTOFF)


def Wprime_image(s):
    """Return the derivative of the M=160 real-image reference potential."""
    return periodized_Wprime(s, LJ_PARAMETERS, REAL_IMAGE_CUTOFF)


def load_atomistic_profiles(path):
    """Load normalized warm-track states through the unsuffixed NPZ aliases."""
    with np.load(path) as source:
        saved_N = tuple(int(value) for value in source["N_values"])
        if saved_N != N_VALUES:
            raise ValueError(
                f"Expected atom counts {N_VALUES}, found {saved_N} in {path}.")
        if int(source["pair_cutoff"]) != 80:
            raise ValueError("The saved atomistic profiles do not use M=80.")
        profiles = {}
        for N in N_VALUES:
            profiles[N] = np.concatenate((
                np.array(source[f"u1_atomistic_N{N}"], copy=True),
                np.array(source[f"u2_atomistic_N{N}"], copy=True),
            ))
    return profiles


def solve_references():
    """Solve K=5 on two meshes and K=6/M=160 on the fine mesh."""
    k5 = {
        points: solve_continuum_reference(
            points, W5, Wprime5, FOURIER_GRADIENT_TOL,
            FOURIER_EL_RESIDUAL_TOL)
        for points in CONTINUUM_POINTS
    }
    fine_points = CONTINUUM_POINTS[-1]
    return {
        "k5": k5,
        "k6": solve_continuum_reference(
            fine_points, W6, Wprime6, FOURIER_GRADIENT_TOL,
            FOURIER_EL_RESIDUAL_TOL),
        "image": solve_continuum_reference(
            fine_points, W_image, Wprime_image, FOURIER_GRADIENT_TOL,
            FOURIER_EL_RESIDUAL_TOL),
    }


def compute_convergence(atomistic_profiles, references):
    """Sample the K=5 reference and compute errors for every atom count."""
    fine_k5 = references["k5"][CONTINUUM_POINTS[-1]]
    spline = periodic_q_spline(fine_k5["x"], fine_k5["q"])
    rows = []
    for N in N_VALUES:
        continuum_pair = sample_continuum_pair(spline, N)
        errors = two_layer_errors(atomistic_profiles[N], continuum_pair, N)
        rows.append({
            "N": N,
            "h": atomistic_geometry(N)[-1],
            "atomistic_u": atomistic_profiles[N],
            "continuum_pair": continuum_pair,
            **errors,
        })

    for index, row in enumerate(rows):
        if index == 0:
            row["h2_adjacent_order"] = np.nan
            row["max_adjacent_order"] = np.nan
            continue
        previous = rows[index - 1]
        denominator = np.log(row["N"] / previous["N"])
        row["h2_adjacent_order"] = float(
            np.log(previous["h2_error"] / row["h2_error"]) / denominator)
        row["max_adjacent_order"] = float(
            np.log(previous["max_error"] / row["max_error"]) / denominator)
    return rows


def reference_changes(references, finest_N):
    """Compare continuum discretization, Fourier cutoff, and image evaluation."""
    splines = {
        "k5_coarse": periodic_q_spline(
            references["k5"][CONTINUUM_POINTS[0]]["x"],
            references["k5"][CONTINUUM_POINTS[0]]["q"],
        ),
        "k5_fine": periodic_q_spline(
            references["k5"][CONTINUUM_POINTS[-1]]["x"],
            references["k5"][CONTINUUM_POINTS[-1]]["q"],
        ),
        "k6": periodic_q_spline(
            references["k6"]["x"], references["k6"]["q"]),
        "image": periodic_q_spline(
            references["image"]["x"], references["image"]["q"]),
    }
    pairs = {name: sample_continuum_pair(spline, finest_N)
             for name, spline in splines.items()}
    return {
        "continuum_mesh": two_layer_errors(
            pairs["k5_coarse"], pairs["k5_fine"], finest_N),
        "fourier_cutoff": two_layer_errors(
            pairs["k5_fine"], pairs["k6"], finest_N),
        "real_image": two_layer_errors(
            pairs["k5_fine"], pairs["image"], finest_N),
    }


def verification_checks(rows, changes):
    """Return potential-level and reference-level acceptance checks."""
    phase = np.linspace(-np.pi, np.pi, 4001)
    k5_k6_W = float(np.max(np.abs(W5(phase) - W6(phase))))
    k5_k6_Wprime = float(
        np.max(np.abs(Wprime5(phase) - Wprime6(phase))))
    k5_image_W = float(np.max(np.abs(W5(phase) - W_image(phase))))
    k5_image_Wprime = float(
        np.max(np.abs(Wprime5(phase) - Wprime_image(phase))))
    finest = rows[-1]
    return [
        ("K=5 to K=6 potential change", k5_k6_W <= 2.0e-11,
         k5_k6_W, "<= 2.0e-11"),
        ("K=5 to K=6 derivative change", k5_k6_Wprime <= 1.1e-10,
         k5_k6_Wprime, "<= 1.1e-10"),
        ("K=5 to M=160 potential change", k5_image_W <= 5.0e-11,
         k5_image_W, "<= 5.0e-11"),
        ("K=5 to M=160 derivative change", k5_image_Wprime <= 1.1e-10,
         k5_image_Wprime, "<= 1.1e-10"),
        ("K=5 to K=6 H_h^2 reference change",
         changes["fourier_cutoff"]["h2_error"] <
         REFINEMENT_FRACTION * finest["h2_error"],
         changes["fourier_cutoff"]["h2_error"],
         f"< 5% of {finest['h2_error']:.6e}"),
        ("K=5 to K=6 maximum reference change",
         changes["fourier_cutoff"]["max_error"] <
         REFINEMENT_FRACTION * finest["max_error"],
         changes["fourier_cutoff"]["max_error"],
         f"< 5% of {finest['max_error']:.6e}"),
        ("K=5 to M=160 H_h^2 reference change",
         changes["real_image"]["h2_error"] <
         REFINEMENT_FRACTION * finest["h2_error"],
         changes["real_image"]["h2_error"],
         f"< 5% of {finest['h2_error']:.6e}"),
        ("K=5 to M=160 maximum reference change",
         changes["real_image"]["max_error"] <
         REFINEMENT_FRACTION * finest["max_error"],
         changes["real_image"]["max_error"],
         f"< 5% of {finest['max_error']:.6e}"),
        ("K=5 continuum 400/800 H_h^2 change",
         changes["continuum_mesh"]["h2_error"] <
         REFINEMENT_FRACTION * finest["h2_error"],
         changes["continuum_mesh"]["h2_error"],
         f"< 5% of {finest['h2_error']:.6e}"),
        ("K=5 continuum 400/800 maximum change",
         changes["continuum_mesh"]["max_error"] <
         REFINEMENT_FRACTION * finest["max_error"],
         changes["continuum_mesh"]["max_error"],
         f"< 5% of {finest['max_error']:.6e}"),
    ]


def run_comparison(source_path):
    """Run the Fourier continuum solves and assemble all comparisons."""
    atomistic_profiles = load_atomistic_profiles(source_path)
    references = solve_references()
    rows = compute_convergence(atomistic_profiles, references)
    changes = reference_changes(references, N_VALUES[-1])
    checks = verification_checks(rows, changes)
    N = [row["N"] for row in rows]
    slopes = {
        "h2_all": loglog_slope(
            N, [row["h2_error"] for row in rows]),
        "maximum_all": loglog_slope(
            N, [row["max_error"] for row in rows]),
    }
    return {
        "source_path": source_path,
        "references": references,
        "rows": rows,
        "changes": changes,
        "checks": checks,
        "slopes": slopes,
    }


def save_summary(outdir, comparison):
    """Write the Fourier-reference convergence table."""
    rows = []
    for result in comparison["rows"]:
        rows.append({key: value for key, value in result.items()
                     if key not in {"atomistic_u", "continuum_pair"}})
    with (outdir / "convergence_summary.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_profiles(outdir, comparison):
    """Save the atomistic, Fourier-continuum, and continuum-reference arrays."""
    arrays = {
        "N_values": np.asarray(N_VALUES),
        "fourier_modes": np.asarray(FOURIER_MODES),
        "fourier_constant": np.asarray(FOURIER_CONSTANT),
        "fourier_coefficients": COEFFICIENTS_5,
    }
    for row in comparison["rows"]:
        N = row["N"]
        atom1, atom2 = split_layers(row["atomistic_u"], N)
        cont1, cont2 = split_layers(row["continuum_pair"], N)
        arrays.update({
            f"u1_atomistic_N{N}": atom1,
            f"u2_atomistic_N{N}": atom2,
            f"u1_fourier_continuum_N{N}": cont1,
            f"u2_fourier_continuum_N{N}": cont2,
        })
    for label, reference in (
            ("k5_400", comparison["references"]["k5"][400]),
            ("k5_800", comparison["references"]["k5"][800]),
            ("k6_800", comparison["references"]["k6"]),
            ("image_800", comparison["references"]["image"])):
        arrays[f"x_{label}"] = reference["x"]
        arrays[f"q_{label}"] = reference["q"]
    np.savez(outdir / "profiles.npz", **arrays)


def save_plot(outdir, comparison):
    """Plot the recomputed convergence errors against N."""
    N = np.asarray([row["N"] for row in comparison["rows"]], dtype=float)
    h2 = np.asarray([row["h2_error"] for row in comparison["rows"]])
    maximum = np.asarray([row["max_error"] for row in comparison["rows"]])
    reference = h2[0] * N[0] / N

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    axis.loglog(N, h2, "o-", label=r"$H_h^2$ error")
    axis.loglog(N, maximum, "s-", label=r"maximum error")
    axis.loglog(N, reference, "--", color="black", label=r"$N^{-1}$ reference")
    axis.set(xlabel="$N$", ylabel="error",
             title="Atomistic convergence to the K=5 LJ-Fourier continuum")
    axis.grid(True, which="both", linestyle="--", alpha=0.5)
    axis.legend()
    figure.tight_layout()
    figure.savefig(outdir / "convergence.png", dpi=150)
    plt.close(figure)


def save_report(outdir, comparison, overall_pass):
    """Write and print the Fourier-continuum verification report."""
    parameters = LJ_PARAMETERS
    lines = [
        "Atomistic LJ / Fourier-continuum convergence comparison",
        "=======================================================",
        "The warm-track atomistic profiles are reused from the M=80 study.",
        "The fitted slopes are empirical evidence, not a proof of global "
        "minimality or of the paper's uniform atomistic stability gap.",
        "",
        "Model",
        f"  source: {comparison['source_path']}",
        f"  a={parameters.lattice_constant:g}, sigma={parameters.sigma:g}, "
        f"L={parameters.interlayer_distance:g}, epsilon={parameters.epsilon:g}",
        f"  continuum Fourier cutoff: K={FOURIER_MODES}",
        "",
        "Convergence",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"  N={row['N']:3d}: H2={row['h2_error']:.6e}, "
            f"max={row['max_error']:.6e}, "
            f"orders=({row['h2_adjacent_order']:.4f}, "
            f"{row['max_adjacent_order']:.4f})")
    slopes = comparison["slopes"]
    lines.extend([
        "",
        "Log-log slopes against N; expected value -1",
        f"  H2, all {len(N_VALUES)}: {slopes['h2_all']:.6f}",
        f"  maximum, all {len(N_VALUES)}: {slopes['maximum_all']:.6f}",
        "",
        f"Reference changes on N={N_VALUES[-1]}",
    ])
    for name, change in comparison["changes"].items():
        lines.append(
            f"  {name}: H2={change['h2_error']:.6e}, "
            f"max={change['max_error']:.6e}")
    lines.extend(["", "Acceptance checks"])
    for name, passed, value, requirement in comparison["checks"]:
        lines.append(f"  {'PASS' if passed else 'FAIL'}  {name}: "
                     f"value={value:.6e}, required {requirement}")
    lines.extend(["", f"OVERALL: {'PASS' if overall_pass else 'FAIL'}"])
    report = "\n".join(lines) + "\n"
    (outdir / "check_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")


def main():
    """Run the postprocessing comparison and save its new artifacts."""
    repository = Path(__file__).resolve().parents[2]
    source = (repository / "outputs" / "relaxation" /
              "output_atomistic_two_chain_lj_lbfgs_convergence_study" /
              "profiles.npz")
    outdir = (repository / "outputs" / "relaxation" /
              "output_atomistic_two_chain_lj_fourier_continuum_convergence_comparison")
    comparison = run_comparison(source)
    overall_pass = all(passed for _, passed, _, _ in comparison["checks"])
    outdir.mkdir(parents=True, exist_ok=True)
    save_summary(outdir, comparison)
    save_profiles(outdir, comparison)
    save_plot(outdir, comparison)
    save_report(outdir, comparison, overall_pass)
    if not overall_pass:
        raise RuntimeError(
            f"Fourier-continuum checks failed; see {outdir / 'check_report.txt'}")


if __name__ == "__main__":
    main()
