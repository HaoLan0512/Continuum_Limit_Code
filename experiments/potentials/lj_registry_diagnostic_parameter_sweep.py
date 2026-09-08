"""Generate a Lennard-Jones registry sweep with minimum-location diagnostics.

Method:
    Monatomic Lennard-Jones 12-6 lattice summation with phase-minimum checks.
Purpose:
    Build W and W', sweep ``(sigma, L)``, and cross-check AA/AB classifications.
Inputs:
    CLI parameters for LJ constants, layer geometry, truncation, phase sampling,
    tolerances, and sweep grid.
Outputs:
    Three W-profile PNGs, an LJ-family PNG, a registry-map PNG, and a sweep CSV.
Output location:
    ``outputs/potentials/output_lj_registry_diagnostic_parameter_sweep`` by default.
Dependencies:
    NumPy and Matplotlib; no local module imports.
Related files:
    Overlaps ``lj_lattice_sum_parameter_sweep.py`` but adds minimum-phase and
    classification-consistency diagnostics with different default parameters.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np

TWOPI = 2.0 * np.pi
PI = np.pi


@dataclass(frozen=True)
class LatticeSumParams:
    epsilon: float = 1.0  # ε in the LJ potential
    sigma: float = 0.9  # σ in the LJ potential
    interlayer_distance: float = 1.0  #L
    lattice_constant: float = 1.0  #a, distance between lattice sites in the first layer
    twist_theta: float = 0.0  # θ in a2 = a(1-θ), distance between lattice sites in the second layer
    image_count: int = 80  # Truncation of the lattice sum
    prefactor_W: float = 2.0  # Use for continuum code as W = 2V

    @property
    def a2(self) -> float:
        return self.lattice_constant * (1.0 - self.twist_theta)  # a2 = a(1-θ)


def validate_params(params: LatticeSumParams) -> None:
    """
    Validate parameters for physical and numerical sanity. Raises ValueError if any check fails.
    """
    if params.epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.")
    if params.sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if params.interlayer_distance <= 0.0:
        raise ValueError("interlayer_distance L must be > 0.")
    if params.lattice_constant <= 0.0:
        raise ValueError("lattice_constant a must be > 0.")
    if not (0.0 <= params.twist_theta < 1.0):
        raise ValueError("twist_theta must satisfy 0 <= theta < 1.")
    if params.image_count < 1:
        raise ValueError("image_count must be >= 1.")
    if params.prefactor_W <= 0.0:
        raise ValueError("prefactor_W must be > 0.")


def lj_12_6(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    """Lennard-Jones 12-6 pair potential."""
    sr6 = (sigma / r)**6
    return 4.0 * epsilon * (sr6**2 - sr6)


def lj_12_6_dr(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    """Derivative dV/dr for LJ 12-6."""
    sr6 = (sigma / r)**6
    return 24.0 * epsilon * (sr6 - 2.0 * sr6**2) / r


class LatticeSummedPotential:
    """Periodicized potential W(phi) built from pair interactions."""

    def __init__(self, params: LatticeSumParams):
        validate_params(params)
        self.params = params
        self._phase_to_shift = params.a2 / TWOPI

        m = np.arange(-params.image_count, params.image_count + 1, dtype=float)
        lattice_sites = params.a2 * m

        self._sites = lattice_sites
        self._L = params.interlayer_distance

    def _phase_to_lateral_shift(self, phase: np.ndarray) -> np.ndarray:
        phase_wrapped = (phase + PI) % TWOPI - PI
        return self._phase_to_shift * phase_wrapped

    def W(self, phase: np.ndarray | float) -> np.ndarray | float:
        """
        Periodic lattice-sum potential.

        W(phi) = C * sum_j V_LJ(r_j),
        r_j = sqrt((x(phi) - m*a2)^2 + L^2), m*a2 are the lattice sites in the second layer,
        x(phi) = (a2 / 2pi) * wrap(phi), with wrap(phi) in [-pi, pi).

        Accepts scalar or vector phase input.
        """
        phase_arr = np.asarray(phase, dtype=float)
        x = self._phase_to_lateral_shift(phase_arr)
        d = np.expand_dims(x, axis=-1) - self._sites
        r = np.sqrt(d * d + self._L * self._L)
        values = lj_12_6(r, self.params.epsilon, self.params.sigma)
        out = self.params.prefactor_W * np.sum(values, axis=-1)
        return float(out) if np.ndim(phase_arr) == 0 else out

    def Wprime(self, phase: np.ndarray | float) -> np.ndarray | float:
        """
        Analytic phase derivative of W (not finite difference).

        W'(phi) = C * (a2 / 2pi) * sum_j [dV/dr(r_j)] * (x - m*a2)/r_j,
        with r_j = sqrt((x - m*a2)^2 + L^2) and x = x(phi).

        The chain rule are as follows: dV/dphi = dV/dr * dr/dd * dd/dx * dx/dphi, 
        = dV/dr * (d/r) * 1 * self._phase_to_shift, where self._phase_to_shift = a2 / (2pi) 

        Accepts scalar or vector phase input.
        """
        phase_arr = np.asarray(phase, dtype=float)
        x = self._phase_to_lateral_shift(phase_arr)
        d = np.expand_dims(x, axis=-1) - self._sites
        r = np.sqrt(d * d + self._L * self._L)

        dV_dr = lj_12_6_dr(r, self.params.epsilon, self.params.sigma)
        dV_dd = dV_dr * (d / r)
        out = self.params.prefactor_W * self._phase_to_shift * np.sum(dV_dd,
                                                                      axis=-1)
        return float(out) if np.ndim(phase_arr) == 0 else out


def circular_distance(a: float, b: float) -> float:
    """
    Return the shortest absolute angular distance |a-b| on a 2*pi-periodic circle.

    Angles are in radians. The result is always in [0, pi], so wrap-around
    cases are handled correctly (e.g. distance between -pi+0.1 and pi-0.1 is 0.2, not ~6.08).
    """
    return float(np.abs(np.angle(np.exp(1j * (a - b)))))


def registry_summary(
    potential: LatticeSummedPotential,
    num_points: int = 4096,
    tol: float = PI / 12.0,
) -> dict[str, float | str]:
    """
    Sample W(phi) on [−π,π), find the lowest sampled point, 
    compute the difference delta = W(-pi) - W(0) to determine registry preference,
    then check whether that phase is near
    0 (AA-like), or
    ±π (AB-like, same point on the circle)
    """
    phase = np.linspace(-PI, PI, num_points, endpoint=False)
    Wvals = potential.W(phase)
    min_idx = int(np.argmin(Wvals))
    min_phase = float(phase[min_idx])
    W0 = float(potential.W(0.0))
    Wn_pi = float(potential.W(-PI))
    delta = Wn_pi - W0

    aa_dist = circular_distance(min_phase, 0.0)
    ab_dist = min(circular_distance(min_phase, -PI),
                  circular_distance(min_phase, PI))
    # If minimum phase is within tol of AA reference -> "AA-like"
    # if within tol of AB reference -> "AB-like"
    # otherwise "off-center / mixed"
    if aa_dist <= tol:
        registry = "AA-like (on-top)"
    elif ab_dist <= tol:
        registry = "AB-like (in-between)"
    else:
        registry = "off-center / mixed"

    return {
        "phi_min": min_phase,
        "W(phi_min)": float(Wvals[min_idx]),
        "W(0)": W0,
        "W(-pi)": Wn_pi,
        "Delta=W(-pi)-W(0)": delta,
        "aa_dist": aa_dist,
        "ab_dist": ab_dist,
        "tol": float(tol),
        "registry": registry,
    }


def assumption_report(potential: LatticeSummedPotential,
                      num_points: int = 4096) -> dict[str, float | bool]:
    """
    Check numerical properties of W(phi) that are assumed in continuum models:
    - Evenness: W(phi) should be even, check max absolute deviation from evenness.
    - Monotonicity: W'(phi) should be negative on (0, pi), check if all sampled points satisfy this.
    - Ratio monotonicity: W'(phi)/phi should be increasing on (0, pi), check if the ratio is non-decreasing at sampled points.
    """
    phase = np.linspace(-PI, PI, num_points, endpoint=False)
    Wvals = potential.W(phase)
    Wvals_neg = potential.W(-phase)
    even_err = float(np.max(np.abs(Wvals - Wvals_neg)))

    Wp = np.gradient(Wvals, phase)
    # Wp = potential.Wprime(phase)   # instead of np.gradient(...)

    mask = (phase > 1.0e-6) & (phase < PI - 1.0e-6)
    decreasing_0_pi = bool(np.all(Wp[mask] < 1.0e-8))

    ratio = Wp[mask] / phase[mask]
    ratio_increasing = bool(np.all(np.diff(ratio) >= -1.0e-4))

    return {
        "max evenness error": even_err,
        "W'(s) < 0 on (0,pi) (numeric)": decreasing_0_pi,
        "(W'(s)/s) increasing on (0,pi) (numeric)": ratio_increasing,
    }


def plot_lj_family(outdir: Path) -> None:
    r = np.linspace(0.65, 4.0, 800)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    eps_fixed = 1.0
    for sigma in (0.7, 0.85, 1.0, 1.15):
        axes[0].plot(r,
                     lj_12_6(r, eps_fixed, sigma),
                     label=f"sigma={sigma:.2f}")
    axes[0].set_title("LJ 12-6: varying sigma (epsilon=1)")
    axes[0].set_xlabel("distance r")
    axes[0].set_ylabel("V_LJ(r)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylim(-1.5, 8.0)
    axes[0].legend()

    sigma_fixed = 1.0
    for epsilon in (0.5, 1.0, 1.5, 2.0):
        axes[1].plot(r,
                     lj_12_6(r, epsilon, sigma_fixed),
                     label=f"epsilon={epsilon:.2f}")
    axes[1].set_title("LJ 12-6: varying epsilon (sigma=1)")
    axes[1].set_xlabel("distance r")
    axes[1].set_ylabel("V_LJ(r)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_ylim(-2.5, 8.0)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "lj_pair_family.png", dpi=180)
    plt.close(fig)


def plot_W_profile(potential: LatticeSummedPotential, outpath: Path,
                   title_prefix: str) -> dict[str, float | str]:
    """
    Plot W(phi) over one period, mark registry points and the global minimum, and return summary info.
    """
    phase = np.linspace(-PI, PI, 4096, endpoint=False)
    Wvals = potential.W(phase)
    summary = registry_summary(potential, num_points=4096)
    phi_min = float(summary["phi_min"])

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.8))

    ax.plot(phase, Wvals, linewidth=2.0, label="W(phi)")
    ax.axvline(0.0,
               linestyle="--",
               color="tab:blue",
               alpha=0.7,
               label="AA (phi=0)")
    ax.axvline(-PI,
               linestyle="--",
               color="tab:orange",
               alpha=0.7,
               label="AB (phi=-pi)")
    ax.plot([phi_min], [float(summary["W(phi_min)"])],
            "o",
            color="black",
            markersize=6,
            label="global min")
    ax.set_xlim(-1.1 * PI, 1.1 * PI)
    ax.set_xlabel("registry phase phi (radians)")
    ax.set_ylabel("W(phi)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    ax.set_title(title_prefix)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return summary


def sweep_registry_map(
    template: LatticeSumParams,
    sigma_values: np.ndarray,
    L_values: np.ndarray,
    outdir: Path,
    registry_num_points: int = 1024,
    delta_tol: float = 1.0e-6,
    exact_phase_tol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep over (sigma, L) parameter space and classify the preferred registry
    using a consistency gate:
      - delta sign test from W(-pi)-W(0)
      - exact minimum-location test from aa_dist/ab_dist
    A cell is AA-like or AB-like only when both tests agree; otherwise degenerate.
    """
    if registry_num_points < 32:
        raise ValueError(
            "registry_num_points must be >= 32 for stable classification.")
    if delta_tol < 0.0:
        raise ValueError("delta_tol must be non-negative.")
    if exact_phase_tol < 0.0:
        raise ValueError("exact_phase_tol must be non-negative.")

    delta = np.zeros((L_values.size, sigma_values.size), dtype=float)
    label = np.empty((L_values.size, sigma_values.size), dtype=int)

    rows: list[list[str]] = []
    rows.append([
        "L",
        "sigma",
        "W(0)",
        "W(-pi)",
        "Delta=W(-pi)-W(0)",
        "phi_min",
        "aa_dist",
        "ab_dist",
        "delta_pref",
        "exact_phase_pref",
        "consistent",
        "preferred",
    ])

    for i, L in enumerate(L_values):
        for j, sigma in enumerate(sigma_values):
            params = replace(template,
                             interlayer_distance=float(L),
                             sigma=float(sigma))
            potential = LatticeSummedPotential(params)
            W0 = float(potential.W(0.0))
            Wn_pi = float(potential.W(-PI))
            d = Wn_pi - W0
            delta[i, j] = d

            if d > delta_tol:
                delta_pref = "AA-like"
            elif d < -delta_tol:
                delta_pref = "AB-like"
            else:
                delta_pref = "degenerate"

            if delta_pref == "degenerate":
                phi_min = np.nan
                aa_dist = np.nan
                ab_dist = np.nan
                minloc_pref = "degenerate"
                consistent = False
                preferred = "degenerate"
                label[i, j] = 2
            else:
                summary = registry_summary(
                    potential,
                    num_points=registry_num_points,
                )
                phi_min = float(summary["phi_min"])
                aa_dist = float(summary["aa_dist"])
                ab_dist = float(summary["ab_dist"])

                if aa_dist <= exact_phase_tol:
                    minloc_pref = "AA-like"
                elif ab_dist <= exact_phase_tol:
                    minloc_pref = "AB-like"
                else:
                    minloc_pref = "degenerate"

                consistent = (delta_pref == minloc_pref)
                if consistent and delta_pref == "AA-like":
                    preferred = "AA-like"
                    label[i, j] = 0
                elif consistent and delta_pref == "AB-like":
                    preferred = "AB-like"
                    label[i, j] = 1
                else:
                    preferred = "degenerate"
                    label[i, j] = 2

            rows.append([
                f"{L:.6g}", f"{sigma:.6g}", f"{W0:.8g}", f"{Wn_pi:.8g}",
                f"{d:.8g}", f"{phi_min:.8g}", f"{aa_dist:.3e}",
                f"{ab_dist:.3e}", delta_pref, minloc_pref,
                "yes" if consistent else "no", preferred
            ])

    csv_path = outdir / "registry_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    # Robust signed-log normalization around 0:
    # cap range at 99th percentile of |Delta| and choose linear window from
    # 60th percentile of |Delta| so near-zero structure is visible.
    finite_vals = delta[np.isfinite(delta)]
    if finite_vals.size > 0:
        abs_vals = np.abs(finite_vals)
        q = float(np.nanpercentile(abs_vals, 99.0))
        linthresh = float(np.nanpercentile(abs_vals, 60.0))
    else:
        q = 1.0
        linthresh = 1.0e-12

    if not np.isfinite(q) or q <= 0.0:
        q = 1.0
    if not np.isfinite(linthresh):
        linthresh = 1.0e-12
    linthresh = float(max(linthresh, 1.0e-12))

    norm = SymLogNorm(
        linthresh=linthresh,
        vmin=-q,
        vmax=q,
        base=10.0,
    )
    X, Y = np.meshgrid(sigma_values, L_values)
    im = ax.pcolormesh(X, Y, delta, cmap="RdYlBu_r", norm=norm, shading="auto")
    cbar = fig.colorbar(im, ax=ax, extend="both")
    cbar.set_label(r"$\Delta = W(-\pi)-W(0)$ (robust symlog percentile scale)")
    ax.contour(
        sigma_values,
        L_values,
        delta,
        levels=[0.0],
        colors="black",
        linewidths=1.5,
    )
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$L$")
    ax.set_title(r"Reference-point contrast map: $\Delta = W(-\pi)-W(0)$; "
                 r"$\Delta>0$ => AA, $\Delta<0$ => AB")
    fig.tight_layout()
    fig.savefig(outdir / "registry_delta_map.png", dpi=180)
    plt.close(fig)

    return delta, label


def pick_representative_params(
    template: LatticeSumParams,
    sigma_values: np.ndarray,
    L_values: np.ndarray,
    delta: np.ndarray,
) -> tuple[LatticeSumParams | None, LatticeSumParams | None]:
    tol = 1.0e-6
    aa_params = None
    ab_params = None

    aa_idx = np.unravel_index(np.argmax(delta), delta.shape)
    if delta[aa_idx] > tol:
        aa_params = replace(
            template,
            interlayer_distance=float(L_values[aa_idx[0]]),
            sigma=float(sigma_values[aa_idx[1]]),
        )

    ab_idx = np.unravel_index(np.argmin(delta), delta.shape)
    if delta[ab_idx] < -tol:
        ab_params = replace(
            template,
            interlayer_distance=float(L_values[ab_idx[0]]),
            sigma=float(sigma_values[ab_idx[1]]),
        )

    return aa_params, ab_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/plot W from atomistic pair potential.")
    parser.add_argument("--epsilon",
                        type=float,
                        default=1.0,
                        help="LJ epsilon.")
    parser.add_argument("--sigma", type=float, default=0.9, help="LJ sigma.")
    parser.add_argument("--L",
                        type=float,
                        default=0.75,
                        help="Interlayer distance L.")
    parser.add_argument("--a",
                        type=float,
                        default=1.0,
                        help="Layer-1 lattice constant a.")
    parser.add_argument("--theta",
                        type=float,
                        default=0.05,
                        help="Lattice mismatch theta in a2=a(1-theta).")
    parser.add_argument("--images",
                        type=int,
                        default=80,
                        help="Lattice image truncation M in sum m=-M..M.")
    parser.add_argument("--prefactor-W",
                        type=float,
                        default=2.0,
                        help="Global prefactor on W.")
    parser.add_argument("--output-dir",
                        type=str,
                        default=None,
                        help="Output directory for plots and csv.")

    parser.add_argument("--sweep-sigma-min",
                        type=float,
                        default=0.55,
                        help="Sweep sigma min.")
    parser.add_argument("--sweep-sigma-max",
                        type=float,
                        default=0.90,
                        help="Sweep sigma max.")
    parser.add_argument("--sweep-L-min",
                        type=float,
                        default=0.75,
                        help="Sweep L min.")
    parser.add_argument("--sweep-L-max",
                        type=float,
                        default=1.35,
                        help="Sweep L max.")
    parser.add_argument("--sweep-grid",
                        type=int,
                        default=121,
                        help="Number of points per sweep axis.")
    parser.add_argument(
        "--sweep-registry-points",
        type=int,
        default=1024,
        help=
        "Phase samples used by min-location registry check at each sweep cell.",
    )
    parser.add_argument(
        "--sweep-exact-phase-tol",
        type=float,
        default=1.0e-12,
        help=
        "Tiny tolerance (radians) for exact AA/AB minimum-phase check in sweep.",
    )
    parser.add_argument(
        "--sweep-delta-tol",
        type=float,
        default=1.0e-6,
        help="Energy tolerance for delta sign test in sweep classification.",
    )
    parser.add_argument("--no-sweep",
                        action="store_true",
                        help="Skip sigma-L sweep.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    outdir = Path(
        args.output_dir
    ) if args.output_dir else (
        repository_root
        / "outputs"
        / "potentials"
        / "output_lj_registry_diagnostic_parameter_sweep"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    params = LatticeSumParams(
        epsilon=args.epsilon,
        sigma=args.sigma,
        interlayer_distance=args.L,
        lattice_constant=args.a,
        twist_theta=args.theta,
        image_count=args.images,
        prefactor_W=args.prefactor_W,
    )
    potential = LatticeSummedPotential(params)

    plot_lj_family(outdir)

    summary = plot_W_profile(
        potential,
        outdir / "W_profile_base.png",
        title_prefix=
        f"W from LJ lattice sum: sigma={args.sigma:.3g}, L={args.L:.3g}",
    )

    report = assumption_report(potential)

    print("Base potential summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.8g}")
        else:
            print(f"  {k}: {v}")

    print("\nAssumption diagnostics (numerical):")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3e}")
        else:
            print(f"  {k}: {v}")

    if args.no_sweep:
        print(f"\nSaved outputs in: {outdir}")
        return

    sigma_values = np.linspace(args.sweep_sigma_min, args.sweep_sigma_max,
                               args.sweep_grid)
    L_values = np.linspace(args.sweep_L_min, args.sweep_L_max, args.sweep_grid)

    delta, _ = sweep_registry_map(
        params,
        sigma_values,
        L_values,
        outdir,
        registry_num_points=args.sweep_registry_points,
        delta_tol=args.sweep_delta_tol,
        exact_phase_tol=args.sweep_exact_phase_tol,
    )
    aa_params, ab_params = pick_representative_params(params, sigma_values,
                                                      L_values, delta)

    if aa_params is not None:
        aa_potential = LatticeSummedPotential(aa_params)
        aa_summary = plot_W_profile(
            aa_potential,
            outdir / "W_profile_representative_AA.png",
            title_prefix=
            ("Representative AA-like regime: "
             f"sigma={aa_params.sigma:.3g}, L={aa_params.interlayer_distance:.3g}"
             ),
        )
        print("\nRepresentative AA-like parameters:")
        print(
            f"  sigma={aa_params.sigma:.6g}, L={aa_params.interlayer_distance:.6g}"
        )
        print(f"  predicted registry: {aa_summary['registry']}")

    if ab_params is not None:
        ab_potential = LatticeSummedPotential(ab_params)
        ab_summary = plot_W_profile(
            ab_potential,
            outdir / "W_profile_representative_AB.png",
            title_prefix=
            ("Representative AB-like regime: "
             f"sigma={ab_params.sigma:.3g}, L={ab_params.interlayer_distance:.3g}"
             ),
        )
        print("\nRepresentative AB-like parameters:")
        print(
            f"  sigma={ab_params.sigma:.6g}, L={ab_params.interlayer_distance:.6g}"
        )
        print(f"  predicted registry: {ab_summary['registry']}")

    print(f"\nSaved outputs in: {outdir}")


if __name__ == "__main__":
    main()
