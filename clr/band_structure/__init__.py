"""Reusable CLR Bloch-band solvers.

Method:
    Finite-difference and finite-difference-symbol/FFT Bloch discretizations.
Purpose:
    Package the active import-safe Bloch eigensolvers.
Inputs:
    Potential samples, periodic grids, epsilon, and Bloch momentum.
Outputs:
    None at package import time.
Output location:
    None.
Dependencies:
    NumPy and SciPy through the contained solver modules.
Related files:
    Band-study runners live in ``experiments.band_structure``.
"""
