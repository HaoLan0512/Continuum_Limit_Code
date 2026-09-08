"""Odd, phase-fixed coordinates for an even periodic grid.

Method:
    Reconstruct a full odd grid vector from its independent negative-half
    values and apply the transpose chain rule to a full-grid gradient.
Purpose:
    Share the odd-variable reduction used by continuum relaxation experiments.
Inputs:
    A reduced vector, an optional even grid size, or a full-grid gradient.
Outputs:
    A full odd periodic vector or its reduced-coordinate gradient.
Output location:
    None.
Dependencies:
    NumPy only.
Related files:
    The phase-fixed, unrestricted-reference, and atomistic-continuum
    relaxation workflows import these functions.
"""

import numpy as np


def build_odd_periodic(free_values, point_count=None):
    """Build the odd periodic vector ``[0, a, 0, -a[::-1]]``.

    ``free_values`` stores the values strictly between ``-pi`` and zero. If
    ``point_count`` is omitted, the full even grid size is inferred from the
    reduced-vector length.
    """
    if point_count is None:
        point_count = 2 * (len(free_values) + 1)
    midpoint = point_count // 2
    values = np.zeros(point_count)
    values[1:midpoint] = free_values
    values[midpoint + 1:] = -free_values[::-1]
    return values


def reduce_odd_gradient(full_gradient):
    """Apply the transpose chain rule for the odd parametrization."""
    midpoint = full_gradient.size // 2
    return (full_gradient[1:midpoint] -
            full_gradient[:midpoint:-1])
