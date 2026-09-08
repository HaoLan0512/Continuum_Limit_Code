"""Generate basis-aware Lennard-Jones stacking-potential sweeps.

Method:
    Periodic Lennard-Jones 12-6 lattice summation with monoatomic or diatomic basis.
Purpose:
    Build W and W', plot LJ families, and classify a ``(sigma, L)`` registry sweep.
Inputs:
    CLI parameters for LJ constants, layer geometry, basis, truncation, and sweep grid.
Outputs:
    Three W-profile PNGs, an LJ-family PNG, a registry-map PNG, and a sweep CSV.
Output location:
    ``outputs/potentials/output_lj_lattice_sum_parameter_sweep`` by default.
Dependencies:
    NumPy and Matplotlib; no local module imports.
Related files:
    The diagnostic monatomic variant is
    ``lj_registry_diagnostic_parameter_sweep.py``; conventions use ``W = 2V``
    by default and are checked independently by
    ``lj_potential_convention_comparison.py``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class LatticeSumParams:
    epsilon: float = 1.0
    sigma: float = 0.9
    interlayer_distance: float = 1.0
    lattice_constant: float = 1.0
    twist_theta: float = 0.0
    image_count: int = 80
    prefactor_W: float = 2.0
    basis_offsets: tuple[float, ...] = (0.0, )
    basis_weights: tuple[float, ...] = (1.0, )

    @property
    def a2(self) -> float:
        return self.lattice_constant * (1.0 - self.twist_theta)


def validate_params(params: LatticeSumParams) -> None:
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
    if len(params.basis_offsets) == 0:
        raise ValueError("basis_offsets must be non-empty.")
    if len(params.basis_offsets) != len(params.basis_weights):
        raise ValueError(
            "basis_offsets and basis_weights must have same length.")


def lj_12_6(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    """Lennard-Jones 12-6 pair potential."""
    sr6 = (sigma / r)**6
    return 4.0 * epsilon * (sr6**2 - sr6)


def lj_12_6_dr(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    """Derivative dV/dr for LJ 12-6."""
    inv_r = 1.0 / r
    sr6 = (sigma * inv_r)**6
    sr12 = sr6**2
    return 24.0 * epsilon * inv_r * (sr6 - 2.0 * sr12)


class LatticeSummedPotential:
    """Periodicized potential W(phi) built from pair interactions."""

    def __init__(self, params: LatticeSumParams):
        validate_params(params)
        self.params = params
        self._phase_to_shift = params.a2 / TWOPI

        basis_offsets = np.asarray(params.basis_offsets, dtype=float)
        basis_weights = np.asarray(params.basis_weights, dtype=float)

        m = np.arange(-params.image_count, params.image_count + 1, dtype=float)
        lattice_sites = params.a2 * m

        self._sites = (lattice_sites[:, None] +
                       basis_offsets[None, :]).reshape(-1)
        self._weights = np.tile(basis_weights, lattice_sites.size)
        self._L = params.interlayer_distance

    def _phase_to_lateral_shift(self, phase: np.ndarray) -> np.ndarray:
        phase_wrapped = np.mod(phase, TWOPI)
        return self._phase_to_shift * phase_wrapped

    def W(self, phase: np.ndarray | float) -> np.ndarray | float:
        phase_arr = np.asarray(phase, dtype=float)
        x = self._phase_to_lateral_shift(phase_arr)
        d = np.expand_dims(x, axis=-1) - self._sites
        r = np.sqrt(d * d + self._L * self._L)
        values = lj_12_6(r, self.params.epsilon, self.params.sigma)
        out = self.params.prefactor_W * np.sum(values * self._weights, axis=-1)
        return float(out) if np.ndim(phase_arr) == 0 else out

    def Wprime(self, phase: np.ndarray | float) -> np.ndarray | float:
        phase_arr = np.asarray(phase, dtype=float)
        x = self._phase_to_lateral_shift(phase_arr)
        d = np.expand_dims(x, axis=-1) - self._sites
        r = np.sqrt(d * d + self._L * self._L)

        dV_dr = lj_12_6_dr(r, self.params.epsilon, self.params.sigma)
        dV_dd = dV_dr * (d / r)
        chain = self._phase_to_shift
        out = self.params.prefactor_W * chain * np.sum(dV_dd * self._weights,
                                                       axis=-1)
        return float(out) if np.ndim(phase_arr) == 0 else out


def circular_distance(a: float, b: float) -> float:
    return float(np.abs(np.angle(np.exp(1j * (a - b)))))


def registry_summary(potential: LatticeSummedPotential,
                     num_points: int = 4096) -> dict[str, float | str]:
    phase = np.linspace(0.0, TWOPI, num_points, endpoint=False)
    Wvals = potential.W(phase)
    min_idx = int(np.argmin(Wvals))
    min_phase = float(phase[min_idx])
    W0 = float(potential.W(0.0))
    Wpi = float(potential.W(np.pi))
    delta = Wpi - W0

    aa_dist = min(circular_distance(min_phase, 0.0),
                  circular_distance(min_phase, TWOPI))
    ab_dist = circular_distance(min_phase, np.pi)
    tol = np.pi / 12.0

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
        "W(pi)": Wpi,
        "Delta=W(pi)-W(0)": delta,
        "registry": registry,
    }


def assumption_report(potential: LatticeSummedPotential,
                      num_points: int = 4096) -> dict[str, float | bool]:
    phase = np.linspace(0.0, TWOPI, num_points, endpoint=False)
    Wvals = potential.W(phase)
    Wvals_neg = potential.W(-phase)
    even_err = float(np.max(np.abs(Wvals - Wvals_neg)))

    Wp = np.gradient(Wvals, phase)
    mask = (phase > 1.0e-6) & (phase < np.pi - 1.0e-6)
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
    phase = np.linspace(0.0, TWOPI, 4096, endpoint=False)
    Wvals = potential.W(phase)
    Wprime = potential.Wprime(phase)
    summary = registry_summary(potential, num_points=4096)
    phi_min = float(summary["phi_min"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(phase, Wvals, linewidth=2.0, label="W(phi)")
    axes[0].axvline(0.0,
                    linestyle="--",
                    color="tab:blue",
                    alpha=0.7,
                    label="AA (phi=0)")
    axes[0].axvline(np.pi,
                    linestyle="--",
                    color="tab:orange",
                    alpha=0.7,
                    label="AB (phi=pi)")
    axes[0].plot([phi_min], [float(summary["W(phi_min)"])],
                 "o",
                 color="black",
                 markersize=6,
                 label="global min")
    axes[0].set_ylabel("W(phi)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best")
    axes[0].set_title(title_prefix)

    axes[1].plot(phase,
                 Wprime,
                 linewidth=2.0,
                 color="tab:green",
                 label="W'(phi)")
    axes[1].axhline(0.0, linestyle="--", color="black", alpha=0.7)
    axes[1].set_xlabel("registry phase phi (radians)")
    axes[1].set_ylabel("W'(phi)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return summary


def sweep_registry_map(
    template: LatticeSumParams,
    sigma_values: np.ndarray,
    L_values: np.ndarray,
    outdir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.zeros((L_values.size, sigma_values.size), dtype=float)
    label = np.empty((L_values.size, sigma_values.size), dtype=int)

    rows: list[list[str]] = []
    rows.append(
        ["L", "sigma", "W(0)", "W(pi)", "Delta=W(pi)-W(0)", "preferred"])

    for i, L in enumerate(L_values):
        for j, sigma in enumerate(sigma_values):
            params = replace(template,
                             interlayer_distance=float(L),
                             sigma=float(sigma))
            potential = LatticeSummedPotential(params)
            W0 = float(potential.W(0.0))
            Wpi = float(potential.W(np.pi))
            d = Wpi - W0
            delta[i, j] = d

            if d > 1.0e-6:
                preferred = "AA-like"
                label[i, j] = 0
            elif d < -1.0e-6:
                preferred = "AB-like"
                label[i, j] = 1
            else:
                preferred = "degenerate"
                label[i, j] = 2

            rows.append([
                f"{L:.6g}", f"{sigma:.6g}", f"{W0:.8g}", f"{Wpi:.8g}",
                f"{d:.8g}", preferred
            ])

    csv_path = outdir / "registry_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    ext = [sigma_values[0], sigma_values[-1], L_values[0], L_values[-1]]
    im = ax.imshow(
        delta,
        origin="lower",
        aspect="auto",
        extent=ext,
        cmap="coolwarm",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta = W(pi) - W(0)")
    ax.contour(
        sigma_values,
        L_values,
        delta,
        levels=[0.0],
        colors="black",
        linewidths=1.5,
    )
    ax.set_xlabel("sigma")
    ax.set_ylabel("L")
    ax.set_title("Registry preference map: Delta > 0 => AA, Delta < 0 => AB")
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
    aa_params = None
    ab_params = None

    aa_idx = np.unravel_index(np.argmax(delta), delta.shape)
    if delta[aa_idx] > 0.0:
        aa_params = replace(
            template,
            interlayer_distance=float(L_values[aa_idx[0]]),
            sigma=float(sigma_values[aa_idx[1]]),
        )

    ab_idx = np.unravel_index(np.argmin(delta), delta.shape)
    if delta[ab_idx] < 0.0:
        ab_params = replace(
            template,
            interlayer_distance=float(L_values[ab_idx[0]]),
            sigma=float(sigma_values[ab_idx[1]]),
        )

    return aa_params, ab_params


def build_basis(basis: str,
                a2: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if basis == "mono":
        return (0.0, ), (1.0, )
    if basis == "diatomic":
        return (0.0, 0.5 * a2), (1.0, 1.0)
    raise ValueError(f"Unsupported basis type: {basis}")


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
                        default=1.0,
                        help="Interlayer distance L.")
    parser.add_argument("--a",
                        type=float,
                        default=1.0,
                        help="Layer-1 lattice constant a.")
    parser.add_argument("--theta",
                        type=float,
                        default=0.0,
                        help="Lattice mismatch theta in a2=a(1-theta).")
    parser.add_argument("--images",
                        type=int,
                        default=80,
                        help="Lattice image truncation M in sum m=-M..M.")
    parser.add_argument("--prefactor-W",
                        type=float,
                        default=2.0,
                        help="Global prefactor on W.")
    parser.add_argument("--basis",
                        choices=("mono", "diatomic"),
                        default="mono",
                        help="Reference-layer basis model.")
    parser.add_argument("--output-dir",
                        type=str,
                        default=None,
                        help="Output directory for plots and csv.")

    parser.add_argument("--sweep-sigma-min",
                        type=float,
                        default=0.7,
                        help="Sweep sigma min.")
    parser.add_argument("--sweep-sigma-max",
                        type=float,
                        default=1.2,
                        help="Sweep sigma max.")
    parser.add_argument("--sweep-L-min",
                        type=float,
                        default=0.7,
                        help="Sweep L min.")
    parser.add_argument("--sweep-L-max",
                        type=float,
                        default=1.6,
                        help="Sweep L max.")
    parser.add_argument("--sweep-grid",
                        type=int,
                        default=41,
                        help="Number of points per sweep axis.")
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
        / "output_lj_lattice_sum_parameter_sweep"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    a2 = args.a * (1.0 - args.theta)
    basis_offsets, basis_weights = build_basis(args.basis, a2)

    params = LatticeSumParams(
        epsilon=args.epsilon,
        sigma=args.sigma,
        interlayer_distance=args.L,
        lattice_constant=args.a,
        twist_theta=args.theta,
        image_count=args.images,
        prefactor_W=args.prefactor_W,
        basis_offsets=basis_offsets,
        basis_weights=basis_weights,
    )
    potential = LatticeSummedPotential(params)

    plot_lj_family(outdir)

    summary = plot_W_profile(
        potential,
        outdir / "W_profile_base.png",
        title_prefix=
        f"W from LJ lattice sum ({args.basis} basis): sigma={args.sigma:.3g}, L={args.L:.3g}",
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

    delta, _ = sweep_registry_map(params, sigma_values, L_values, outdir)
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
