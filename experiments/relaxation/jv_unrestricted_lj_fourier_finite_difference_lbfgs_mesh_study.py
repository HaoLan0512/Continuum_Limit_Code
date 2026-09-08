"""Check Theorem 1 numerically with a pair-derived LJ Fourier potential.

Method:
    Analytic Poisson-summed LJ Fourier coefficients with five retained modes,
    forward periodic finite differences, unrestricted multistart, phase
    alignment, an odd reference, and L-BFGS-B optimization.
Purpose:
    Repeat the unrestricted mesh study with the paper-matched continuum
    potential W=4*sum_m V(s-2*pi*m), without changing the cosine study.
Inputs:
    N=100, 200, 400; effective LJ parameters a=1, sigma=0.9, L=1,
    epsilon=0.5; Fourier modes k=0,...,5.
Outputs:
    CSV diagnostics, NPZ profiles, two PNG comparisons, and a check report.
Output location:
    outputs/relaxation/
    output_jv_unrestricted_lj_fourier_finite_difference_lbfgs_mesh_study.
Dependencies:
    clr.potentials.lj_periodic and experiments.relaxation.jv_unrestricted_finite_difference_lbfgs_mesh_study
Related files:
    jv_unrestricted_finite_difference_lbfgs_mesh_study.py retains W=2*cos;
    atomistic_two_chain_lj_lbfgs_convergence_study.py uses the same pair V.
"""

import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from clr.potentials.lj_periodic import (
    EffectiveLJParameters,
    fourier_coefficients,
    fourier_W,
    fourier_Wprime,
)
from experiments.relaxation.jv_unrestricted_finite_difference_lbfgs_mesh_study import (
    run_and_save, )

FOURIER_MODES = 5
LJ_GRADIENT_TOL = 3.0e-7
LJ_EL_RESIDUAL_TOL = 2.0e-5
PARAMETERS = EffectiveLJParameters(
    epsilon=0.5,
    sigma=0.9,
    interlayer_distance=1.0,
    lattice_constant=1.0,
)
FOURIER_CONSTANT, FOURIER_COEFFICIENTS = fourier_coefficients(
    PARAMETERS, FOURIER_MODES)


def W(s):
    """Return the five-mode pair-derived continuum potential."""
    return fourier_W(s, FOURIER_CONSTANT, FOURIER_COEFFICIENTS)


def Wprime(s):
    """Return the analytic derivative of the five-mode potential."""
    return fourier_Wprime(s, FOURIER_COEFFICIENTS)


def potential_checks():
    """Return cutoff, symmetry, shape, and continuum-stability checks."""
    phase = np.linspace(-np.pi, np.pi, 4001)
    positive = np.linspace(np.pi / 1000.0, np.pi - np.pi / 1000.0, 999)
    _, reference_coefficients = fourier_coefficients(PARAMETERS, 20)
    W6 = lambda s: fourier_W(s, FOURIER_CONSTANT,
                             reference_coefficients[:FOURIER_MODES + 1])
    Wprime6 = lambda s: fourier_Wprime(
        s, reference_coefficients[:FOURIER_MODES + 1])

    value_change = float(np.max(np.abs(W(phase) - W6(phase))))
    derivative_change = float(np.max(np.abs(Wprime(phase) - Wprime6(phase))))
    evenness = float(np.max(np.abs(W(phase) - W(-phase))))
    oddness = float(np.max(np.abs(Wprime(phase) + Wprime(-phase))))
    positive_derivative = Wprime(positive)
    ratio = positive_derivative / positive
    ratio_increase_defect = float(max(0.0, -np.min(np.diff(ratio))))
    modes = np.arange(1, reference_coefficients.size + 1, dtype=float)
    second_derivative_bound = float(
        np.sum(modes**2 * np.abs(reference_coefficients)))

    return [
        ("K=5 to K=6 potential change", value_change
         <= 2.0e-11, value_change, "<= 2.0e-11"),
        ("K=5 to K=6 derivative change", derivative_change
         <= 1.1e-10, derivative_change, "<= 1.1e-10"),
        ("LJ-Fourier W evenness", evenness <= 1.0e-14, evenness, "<= 1.0e-14"),
        ("LJ-Fourier W' oddness", oddness <= 1.0e-14, oddness, "<= 1.0e-14"),
        ("LJ-Fourier W'<0 on (0,pi)", np.max(positive_derivative)
         < 0.0, float(np.max(positive_derivative)), "< 0"),
        ("LJ-Fourier W'/s increasing", ratio_increase_defect
         <= 1.0e-12, ratio_increase_defect, "<= 1.0e-12"),
        ("Fourier coefficient bound for Lip(W')", second_derivative_bound
         < 1.0, second_derivative_bound, "< 1"),
    ]


def main():
    """Run and save the LJ-Fourier unrestricted mesh study."""
    outdir = (
        Path(__file__).resolve().parents[2] / "outputs" / "relaxation" /
        "output_jv_unrestricted_lj_fourier_finite_difference_lbfgs_mesh_study")
    description = (
        "W=4 sum_m V(s-2*pi*m), analytic LJ Fourier truncation K=5; "
        "a=1, sigma=0.9, L=1, epsilon=0.5; gradient tolerance=3e-7, "
        "EL tolerance=2e-5")
    run_and_save(W, Wprime, outdir, description, potential_checks(),
                 LJ_GRADIENT_TOL, LJ_EL_RESIDUAL_TOL)


if __name__ == "__main__":
    main()
