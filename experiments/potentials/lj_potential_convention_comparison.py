"""Compare two independently implemented Lennard-Jones stacking conventions.

Method:
    Matched-parameter lattice sums and analytical derivatives for two conventions.
Purpose:
    Compare W, W', the ``W = 2V`` scaling, and AA/AB classifications.
Inputs:
    Fixed matched model parameters and a fixed interlayer-distance sweep.
Outputs:
    ``comparison_report.txt``.
Output location:
    ``outputs/potentials/output_lj_potential_convention_comparison``.
Dependencies:
    NumPy; both compared formulations are deliberately implemented locally.
Related files:
    Reproduces conventions from ``lj_lattice_sum_parameter_sweep.py`` and
    ``lj_stacking_potential_exploration.py`` without importing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


TWOPI = 2.0 * np.pi


def lj_12_6(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * (sr6**2 - sr6)


def lj_12_6_dr(r: np.ndarray, epsilon: float, sigma: float) -> np.ndarray:
    sr6 = (sigma / r) ** 6
    return 24.0 * epsilon * (sigma**6) / (r**7) * (1.0 - 2.0 * sr6)


# -------- model A: lj_lattice_sum_parameter_sweep.py -------- #


@dataclass(frozen=True)
class ModelAParams:
    epsilon: float = 1.0
    sigma: float = 1.0
    L: float = 0.9
    a: float = 1.0
    theta: float = 0.0
    images: int = 100
    prefactor_W: float = 1.0
    basis_offsets: tuple[float, ...] = (0.0,)
    basis_weights: tuple[float, ...] = (1.0,)

    @property
    def a2(self) -> float:
        return self.a * (1.0 - self.theta)


def model_a_W(phase: np.ndarray, p: ModelAParams) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    x = p.a2 * np.mod(phase, TWOPI) / TWOPI
    m = np.arange(-p.images, p.images + 1, dtype=float)
    sites = (p.a2 * m[:, None] + np.asarray(p.basis_offsets)[None, :]).reshape(-1)
    weights = np.tile(np.asarray(p.basis_weights), m.size)
    d = x[..., None] - sites
    r = np.sqrt(d**2 + p.L**2)
    return p.prefactor_W * np.sum(lj_12_6(r, p.epsilon, p.sigma) * weights, axis=-1)


def model_a_Wprime(phase: np.ndarray, p: ModelAParams) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    x = p.a2 * np.mod(phase, TWOPI) / TWOPI
    m = np.arange(-p.images, p.images + 1, dtype=float)
    sites = (p.a2 * m[:, None] + np.asarray(p.basis_offsets)[None, :]).reshape(-1)
    weights = np.tile(np.asarray(p.basis_weights), m.size)
    d = x[..., None] - sites
    r = np.sqrt(d**2 + p.L**2)
    dV_dd = lj_12_6_dr(r, p.epsilon, p.sigma) * d / r
    chain = p.a2 / TWOPI
    return p.prefactor_W * chain * np.sum(dV_dd * weights, axis=-1)


# -------- model B: lj_stacking_potential_exploration.py -------- #


@dataclass(frozen=True)
class ModelBParams:
    epsilon: float = 1.0
    sigma: float = 1.0
    L: float = 0.9
    a: float = 1.0
    nsum: int = 100
    cell_shift: float = 0.0


def model_b_V_delta(delta: np.ndarray, p: ModelBParams) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    n = np.arange(-p.nsum, p.nsum + 1, dtype=float)
    dx = delta[..., None] - p.a * n
    r = np.sqrt(dx**2 + p.L**2)
    return np.sum(lj_12_6(r, p.epsilon, p.sigma), axis=-1)


def model_b_dV_ddelta(delta: np.ndarray, p: ModelBParams) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    n = np.arange(-p.nsum, p.nsum + 1, dtype=float)
    dx = delta[..., None] - p.a * n
    r = np.sqrt(dx**2 + p.L**2)
    return np.sum(lj_12_6_dr(r, p.epsilon, p.sigma) * dx / r, axis=-1)


def model_b_phase_to_registry(phase: np.ndarray, p: ModelBParams) -> np.ndarray:
    raw = p.a * np.asarray(phase, dtype=float) / TWOPI + p.cell_shift
    return np.mod(raw, p.a)


def model_b_W_raw(phase: np.ndarray, p: ModelBParams) -> np.ndarray:
    delta = model_b_phase_to_registry(phase, p)
    return model_b_V_delta(delta, p)


def model_b_Wprime_raw(phase: np.ndarray, p: ModelBParams) -> np.ndarray:
    delta = model_b_phase_to_registry(phase, p)
    return (p.a / TWOPI) * model_b_dV_ddelta(delta, p)


# ---------------- comparison utilities ---------------- #


def classify_by_AA_AB(E_AA: float, E_AB: float, tol: float = 1.0e-10) -> str:
    if E_AA < E_AB - tol:
        return "AA"
    if E_AB < E_AA - tol:
        return "AB"
    return "degenerate"


def run_consistency_checks(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Matched settings for direct apples-to-apples check.
    pa = ModelAParams(
        epsilon=1.0,
        sigma=1.0,
        L=0.9,
        a=1.0,
        theta=0.0,
        images=100,
        prefactor_W=1.0,
        basis_offsets=(0.0,),
        basis_weights=(1.0,),
    )
    pb = ModelBParams(
        epsilon=1.0,
        sigma=1.0,
        L=0.9,
        a=1.0,
        nsum=100,
        cell_shift=0.0,
    )

    phase = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
    Wa = model_a_W(phase, pa)
    Wb = model_b_W_raw(phase, pb)
    dWa = model_a_Wprime(phase, pa)
    dWb = model_b_Wprime_raw(phase, pb)

    max_abs_W = float(np.max(np.abs(Wa - Wb)))
    rms_W = float(np.sqrt(np.mean((Wa - Wb) ** 2)))
    max_abs_dW = float(np.max(np.abs(dWa - dWb)))
    rms_dW = float(np.sqrt(np.mean((dWa - dWb) ** 2)))

    # Also test the scale convention difference (model A default prefactor_W=2).
    pa2 = ModelAParams(**{**pa.__dict__, "prefactor_W": 2.0})
    Wa2 = model_a_W(phase, pa2)
    max_abs_scale = float(np.max(np.abs(Wa2 - 2.0 * Wb)))

    lines = []
    lines.append("Matched-parameter consistency check")
    lines.append("----------------------------------")
    lines.append(f"max|W_A - W_B|        = {max_abs_W:.12e}")
    lines.append(f"rms(W_A - W_B)        = {rms_W:.12e}")
    lines.append(f"max|W'_A - W'_B|      = {max_abs_dW:.12e}")
    lines.append(f"rms(W'_A - W'_B)      = {rms_dW:.12e}")
    lines.append(f"max|W_A(pref=2)-2W_B| = {max_abs_scale:.12e}")
    lines.append("")

    # Compare AA/AB preference on a sweep of L values.
    L_values = [0.8, 0.9, 1.0, 1.1, 1.2, 1.27, 1.35, 1.4, 1.5]
    lines.append("AA/AB preference check vs L")
    lines.append("---------------------------")
    lines.append("L       E_AA(A)         E_AB(A)         pref(A)    E_AA(B)         E_AB(B)         pref(B)")

    all_match = True
    for L in L_values:
        paL = ModelAParams(**{**pa.__dict__, "L": L})
        pbL = ModelBParams(**{**pb.__dict__, "L": L})

        E_AA_A = float(model_a_W(np.array([0.0]), paL)[0])
        E_AB_A = float(model_a_W(np.array([np.pi]), paL)[0])
        pref_A = classify_by_AA_AB(E_AA_A, E_AB_A)

        E_AA_B = float(model_b_V_delta(np.array([0.0]), pbL)[0])
        E_AB_B = float(model_b_V_delta(np.array([0.5 * pbL.a]), pbL)[0])
        pref_B = classify_by_AA_AB(E_AA_B, E_AB_B)

        if pref_A != pref_B:
            all_match = False

        lines.append(
            f"{L:4.2f}   {E_AA_A:14.8f}  {E_AB_A:14.8f}  {pref_A:7s}   "
            f"{E_AA_B:14.8f}  {E_AB_B:14.8f}  {pref_B:7s}"
        )

    lines.append("")
    lines.append(f"AA/AB preference agreement across sweep: {all_match}")

    report_path = outdir / "comparison_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[2]
    run_consistency_checks(
        repository_root
        / "outputs"
        / "potentials"
        / "output_lj_potential_convention_comparison"
    )
