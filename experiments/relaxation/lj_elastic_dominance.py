"""Conservative numerical elastic-dominance bounds for the finite LJ model.

The continuum diagnostic bounds ``lambda*sup|W''|`` on ``[-pi, pi]``.
The atomistic diagnostic bounds the nonlinear force Jacobian on the straight
segment between a sampled continuum reference and an atomistic endpoint.
These are sufficient-condition diagnostics, not instability tests or rigorous
interval-arithmetic certificates. A finite collection of grids cannot establish
a bound uniform in every N. All inputs are explicit and import has no effects.
"""

from functools import lru_cache

import numpy as np

from clr.potentials.lj_periodic import pair_potential_second


TWOPI = 2.0 * np.pi
ROUNDING_MARGIN = 64.0 * np.finfo(float).eps


@lru_cache(maxsize=16)
def _second_derivative_stationary_points(parameters):
    """Find all real stationary phases of V'' using the quartic for V'''.

    Put ``z=(c*s/L)^2``, ``c=a/(2*pi)``, and ``p=(sigma/L)^6``.
    Apart from ``s=0``, V''' vanishes at the nonnegative roots of
    ``14*z^4+36*z^3+24*z^2-(4+91*p)*z-6+21*p``.
    Numerical polynomial roots and floating point padding are used; this is
    not validated interval arithmetic.
    """
    distance = parameters.interlayer_distance
    p = (parameters.sigma / distance)**6
    roots = np.polynomial.polynomial.polyroots(
        [-6.0 + 21.0 * p, -4.0 - 91.0 * p, 24.0, 36.0, 14.0])
    real_nonnegative = [
        max(0.0, float(root.real)) for root in roots
        if abs(root.imag) <= 1e-10 * max(1.0, abs(root.real))
        and root.real >= -1e-12
    ]
    positive = distance * TWOPI / parameters.lattice_constant * np.sqrt(
        real_nonnegative)
    return np.unique(np.r_[-positive, 0.0, positive])


def pair_second_bounds(lower, upper, parameters):
    """Return lower/upper bounds for V'' on broadcastable closed intervals.

    Evaluate each endpoint and every enclosed real stationary point. The
    returned arrays have the broadcast shape of ``lower`` and ``upper``.
    A small outward floating point margin is applied after evaluating extrema.
    Inputs must be finite and ordered; bounds are for the unscaled pair law.
    """
    lower, upper = np.broadcast_arrays(np.asarray(lower, dtype=float),
                                       np.asarray(upper, dtype=float))
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("Pair-argument interval endpoints must be finite.")
    if np.any(lower > upper):
        raise ValueError("Pair-argument intervals must satisfy lower <= upper.")
    first = pair_potential_second(lower, parameters)
    last = pair_potential_second(upper, parameters)
    minimum, maximum = np.minimum(first, last), np.maximum(first, last)
    for point in _second_derivative_stationary_points(parameters):
        enclosed = (lower <= point) & (point <= upper)
        value = pair_potential_second(point, parameters)
        minimum = np.where(enclosed, np.minimum(minimum, value), minimum)
        maximum = np.where(enclosed, np.maximum(maximum, value), maximum)
    margin = ROUNDING_MARGIN * np.maximum(np.abs(minimum), np.abs(maximum))
    return (np.nextafter(minimum - margin, -np.inf),
            np.nextafter(maximum + margin, np.inf))


def _check_scale_cutoff(interaction_scale, cutoff):
    if not np.isfinite(interaction_scale) or interaction_scale <= 0.0:
        raise ValueError("interaction_scale must be positive and finite.")
    if not isinstance(cutoff, (int, np.integer)) or cutoff < 0:
        raise ValueError("cutoff must be a nonnegative integer.")


def continuum_regime(parameters, interaction_scale=1.0, cutoff=80,
                     interval_count=2048):
    """Bound ``rho=lambda*sup|W''|`` on the finite-sum phase cell.

    Since ``W(s)=4*sum_m V(s-2*pi*m)``, signed lower/upper image bounds
    are summed on each phase interval before taking the absolute maximum.
    The caller must separately verify that its profile's W arguments lie
    within the reported phase cell; a finite image sum is not exactly periodic.
    ``Lbound`` is the unscaled curvature bound and ``rho=lambda*Lbound``.
    """
    _check_scale_cutoff(interaction_scale, cutoff)
    if not isinstance(interval_count, (int, np.integer)) or interval_count < 1:
        raise ValueError("interval_count must be a positive integer.")
    edges = np.linspace(-np.pi, np.pi, interval_count + 1)
    images = TWOPI * np.arange(-cutoff, cutoff + 1)[None, :]
    lower, upper = pair_second_bounds(edges[:-1, None] - images,
                                      edges[1:, None] - images, parameters)
    # Preserve cancellation between signed image contributions, then pad the
    # floating point sums in proportion to their absolute summands.
    summation_margin = ROUNDING_MARGIN * np.sum(
        np.maximum(np.abs(lower), np.abs(upper)), axis=1)
    lower_sum = np.sum(lower, axis=1) - summation_margin
    upper_sum = np.sum(upper, axis=1) + summation_margin
    curvature_bound = float(4.0 * np.max(
        np.maximum(np.abs(lower_sum), np.abs(upper_sum))))
    rho = float(interaction_scale * curvature_bound)
    return {
        "available": True,
        "rho": rho,
        "gap": 1.0 - rho,
        "established": bool(rho < 1.0),
        "Lbound": curvature_bound,
        "unscaled_curvature_bound": curvature_bound,
        "interaction_scale": float(interaction_scale),
        "cutoff": int(cutoff),
        "interval_count": int(interval_count),
        "phase_cell_lower": -float(np.pi),
        "phase_cell_upper": float(np.pi),
        "bound_method": "signed image intervals; analytic V''' quartic roots",
    }


def atomistic_regime(reference, endpoint, N, parameters, interaction_scale=1.0,
                     cutoff=80):
    """Bound the force Lipschitz constant on a full-state comparison segment.

    States have N and N+1 layer values, concatenated into length 2*N+1.
    For each directed pair from the analytic energy Hessian, its argument A
    varies affinely along the segment. Bound ``|V''(A)|`` on that interval.
    A pair contributes ``hi*lambda*V''(A)*(alpha*e_i-beta*e_j)^2`` to the
    Hessian, with ``alpha=h/ho`` and ``beta=h/hi``. Accumulate absolute row
    sums after mass scaling by ``D_h**(-1/2)`` on both sides. This bounds
    ``||D_h**(-1) H_interaction||`` in the weighted Euclidean norm.

    Report ``rho=C_P*L``, where ``C_P=h1^2/(4*sin(h1/2)^2)``. Poincare
    controls the separately mean-zero layer components, not the relative
    constant mode allowed by the mean-sum gauge. Layer means must therefore
    be diagnosed separately; rho<1 alone does not certify full stability.
    Nonfinite states return unavailable diagnostics rather than a condition
    established claim. Invalid dimensions raise ValueError.
    """
    _check_scale_cutoff(interaction_scale, cutoff)
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer at least two.")
    reference = np.asarray(reference, dtype=float)
    endpoint = np.asarray(endpoint, dtype=float)
    if reference.shape != (2 * N + 1,) or endpoint.shape != (2 * N + 1,):
        raise ValueError(f"Both states must have shape ({2 * N + 1},).")
    h1, h2 = TWOPI / N, TWOPI / (N + 1)
    h = 2.0 * h1 * h2 / (h1 + h2)
    poincare_constant = float(h1**2 / (4.0 * np.sin(h1 / 2.0)**2))
    result = {
        "available": False,
        "rho": float("nan"),
        "gap": float("nan"),
        "established": False,
        "force_lipschitz_bound": float("nan"),
        "poincare_constant": poincare_constant,
        "interaction_scale": float(interaction_scale),
        "cutoff": int(cutoff),
        "N": int(N),
        "bound_method": "mass-scaled absolute Hessian row sums over segment",
    }
    if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(endpoint))):
        result["reason"] = "Nonfinite reference or endpoint."
        return result
    reference_layers = (reference[:N], reference[N:])
    endpoint_layers = (endpoint[:N], endpoint[N:])
    row_sums = np.zeros(2 * N + 1)
    images = np.arange(-cutoff, cutoff + 1)[None, :]
    layer_data = (
        (0, 1, N, N + 1, h1, h2, 1.0, 0, N),
        (1, 0, N + 1, N, h2, h1, -1.0, N, 0),
    )
    for layer, other, Ni, No, hi, ho, sign, offset, other_offset in layer_data:
        n = np.arange(Ni)[:, None]
        opposite_indices = (n + images) % No
        alpha, beta = h / ho, h / hi
        base = sign * h * n - TWOPI * beta * images
        first = (base + alpha * reference_layers[layer][:, None]
                 - beta * reference_layers[other][opposite_indices])
        last = (base + alpha * endpoint_layers[layer][:, None]
                - beta * endpoint_layers[other][opposite_indices])
        lower, upper = pair_second_bounds(np.minimum(first, last),
                                          np.maximum(first, last), parameters)
        weights = hi * interaction_scale * np.maximum(np.abs(lower),
                                                       np.abs(upper))
        cross_coefficient = alpha * beta / np.sqrt(hi * ho)
        row_sums[offset:offset + Ni] += np.sum(
            weights * (alpha**2 / hi + cross_coefficient), axis=1)
        np.add.at(row_sums, (opposite_indices + other_offset).ravel(),
                  (weights * (beta**2 / ho + cross_coefficient)).ravel())
    force_bound = float(np.nextafter(np.max(row_sums) *
                                    (1.0 + ROUNDING_MARGIN), np.inf))
    rho = poincare_constant * force_bound
    result.update(available=True, rho=rho, gap=1.0 - rho,
                  established=bool(rho < 1.0),
                  force_lipschitz_bound=force_bound)
    return result
