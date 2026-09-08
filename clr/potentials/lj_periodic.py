"""Pair-derived periodic Lennard-Jones continuum potential.

Method:
    Poisson summation of the effective one-dimensional Lennard-Jones pair
    potential at fixed interlayer distance, with an independent real-image sum.
Purpose:
    Evaluate the paper-matched pair potential V and continuum potential
    W(s)=4*sum_m V(s-2*pi*m) either in real space or with a Fourier cutoff.
Inputs:
    Effective LJ parameters, phase values, and an image or Fourier cutoff.
Outputs:
    Potential values, derivatives, and analytic cosine coefficients.
Output location:
    None.
Dependencies:
    NumPy only.
Related files:
    The atomistic convergence study uses the same V in its discrete energy;
    the LJ-Fourier relaxation studies use the analytic coefficients below.
"""

from dataclasses import dataclass

import numpy as np


TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class EffectiveLJParameters:
    """Physical parameters for the smooth effective pair potential.

    Notation-to-code map
    --------------------
    =====================  =======================
    Mathematical notation  Dataclass field
    =====================  =======================
    ``epsilon``             ``epsilon``
    ``sigma``               ``sigma``
    ``L``                   ``interlayer_distance``
    ``a``                   ``lattice_constant``
    =====================  =======================

    Here ``L`` is the fixed distance between the two layers and ``a`` is the
    physical lattice period corresponding to a phase period of ``2*pi``.
    """

    epsilon: float = 0.5
    sigma: float = 0.9
    interlayer_distance: float = 1.0
    lattice_constant: float = 1.0

    def __post_init__(self):
        if min(self.epsilon, self.sigma, self.interlayer_distance,
               self.lattice_constant) <= 0.0:
            raise ValueError("All effective Lennard-Jones parameters must be positive.")


def pair_potential(s, parameters):
    """Return the even effective LJ interaction ``V(s)``.

    Mathematical construction
    -------------------------
    With ``c = a/(2*pi)`` and ``R_2(s) = L^2 + c^2*s^2``, define

    ``S_6(s) = sigma^6/R_2(s)^3``

    and

    ``V(s) = 4*epsilon*(S_6(s)^2 - S_6(s))``.

    Equivalently,

    ``V(s) = 4*epsilon*[sigma^12/R_2(s)^6 - sigma^6/R_2(s)^3]``.

    Notation-to-code map
    --------------------
    =====================  ===========================================
    Mathematical notation  Code variable
    =====================  ===========================================
    ``s``                   ``s``; converted to the array ``phase``
    ``epsilon``             ``parameters.epsilon``
    ``sigma``               ``parameters.sigma``
    ``L``                   ``parameters.interlayer_distance``
    ``a``                   ``parameters.lattice_constant``
    ``c``                   ``scale``
    ``R_2(s)``              ``radius2``
    ``S_6(s)``              ``sr6``
    ``V(s)``                returned scalar or NumPy array
    =====================  ===========================================
    """
    phase = np.asarray(s, dtype=float)
    scale = parameters.lattice_constant / TWOPI
    radius2 = (scale * phase)**2 + parameters.interlayer_distance**2
    sr6 = parameters.sigma**6 / radius2**3
    return 4.0 * parameters.epsilon * (sr6**2 - sr6)


def pair_potential_prime(s, parameters):
    """Return the phase derivative ``V'(s)``.

    Mathematical construction
    -------------------------
    For ``c``, ``R_2``, and ``S_6`` defined as in :func:`pair_potential`,

    ``V'(s) = 24*epsilon*c^2*s*(S_6(s) - 2*S_6(s)^2)/R_2(s)``.

    Notation-to-code map
    --------------------
    =====================  ===========================================
    Mathematical notation  Code variable
    =====================  ===========================================
    ``s``                   ``s``; converted to the array ``phase``
    ``epsilon``             ``parameters.epsilon``
    ``sigma``               ``parameters.sigma``
    ``L``                   ``parameters.interlayer_distance``
    ``a``                   ``parameters.lattice_constant``
    ``c``                   ``scale``
    ``R_2(s)``              ``radius2``
    ``S_6(s)``              ``sr6``
    ``V'(s)``               returned scalar or NumPy array
    =====================  ===========================================
    """
    phase = np.asarray(s, dtype=float)
    scale = parameters.lattice_constant / TWOPI
    radius2 = (scale * phase)**2 + parameters.interlayer_distance**2
    sr6 = parameters.sigma**6 / radius2**3
    return (24.0 * parameters.epsilon * scale**2 * phase *
            (sr6 - 2.0 * sr6**2) / radius2)


def pair_potential_second(s, parameters):
    """Return the second phase derivative ``V''(s)``.

    Mathematical construction
    -------------------------
    For ``c``, ``R_2``, and ``S_6`` defined as in :func:`pair_potential`, let

    ``F_1(s) = 1/R_2(s) - 2*c^2*s^2/R_2(s)^2``

    and

    ``F_2(s) = -6*c^2*s^2*S_6(s)*(1 - 4*S_6(s))/R_2(s)^2``.

    Then

    ``V''(s) = 24*epsilon*c^2*``
    ``[F_1(s)*(S_6(s) - 2*S_6(s)^2) + F_2(s)]``.

    Notation-to-code map
    --------------------
    =====================  ===========================================
    Mathematical notation  Code variable
    =====================  ===========================================
    ``s``                   ``s``; converted to the array ``phase``
    ``epsilon``             ``parameters.epsilon``
    ``sigma``               ``parameters.sigma``
    ``L``                   ``parameters.interlayer_distance``
    ``a``                   ``parameters.lattice_constant``
    ``c^2``                 ``scale2``
    ``R_2(s)``              ``radius2``
    ``S_6(s)``              ``sr6``
    ``F_1(s)``              ``first_factor``
    ``F_2(s)``              ``second_factor``
    ``V''(s)``              returned scalar or NumPy array
    =====================  ===========================================
    """
    phase = np.asarray(s, dtype=float)
    scale2 = (parameters.lattice_constant / TWOPI)**2
    radius2 = scale2 * phase**2 + parameters.interlayer_distance**2
    sr6 = parameters.sigma**6 / radius2**3
    first_factor = 1.0 / radius2 - 2.0 * scale2 * phase**2 / radius2**2
    second_factor = (-6.0 * scale2 * phase**2 * sr6 *
                     (1.0 - 4.0 * sr6) / radius2**2)
    return 24.0 * parameters.epsilon * scale2 * (
        first_factor * (sr6 - 2.0 * sr6**2) + second_factor)


def _image_arguments(s, image_count):
    """Construct the arguments ``x_m(s) = s - 2*pi*m``.

    Mathematical construction
    -------------------------
    For a symmetric real-image cutoff ``M``, this function returns

    ``(x_m(s))_{m=-M}^M = (s - 2*pi*m)_{m=-M}^M``.

    The returned array has one new final axis of length ``2*M + 1``. That axis
    enumerates the image index ``m`` for every supplied value of ``s``.

    Notation-to-code map
    --------------------
    =====================  ===========================================
    Mathematical notation  Code variable
    =====================  ===========================================
    ``s``                   ``s``; converted to the array ``phase``
    ``m``                   entries of ``images``
    ``M``                   ``image_count``
    ``2*pi``                ``TWOPI``
    ``x_m(s)``              returned array
    =====================  ===========================================
    """
    if image_count < 0:
        raise ValueError("image_count must be nonnegative.")
    phase = np.asarray(s, dtype=float)
    images = np.arange(-image_count, image_count + 1, dtype=float)
    return phase[..., None] - TWOPI * images


def periodized_W(s, parameters, image_count):
    """Return the symmetric real-image approximation to ``W(s)``.

    Mathematical construction
    -------------------------
    Following the paper's convention, the infinite sums are

    ``Phi(s) = 2*sum_{m in Z} V(s - 2*pi*m)``

    and ``W(s) = 2*Phi(s)``. With real-image cutoff ``M``, this function uses

    ``Phi_M(s) = 2*sum_{m=-M}^M V(s - 2*pi*m)``,

    ``W_M(s) = 2*Phi_M(s) = 4*sum_{m=-M}^M V(s - 2*pi*m)``.

    Thus ``M`` truncates the image index; it is not a cutoff on the size of an
    individual summand.

    Notation-to-code map
    --------------------
    =====================  ================================================
    Mathematical notation  Code variable
    =====================  ================================================
    ``s``                   ``s``
    ``m``                   final-axis index made by ``_image_arguments``
    ``M``                   ``image_count``
    ``V``                   ``pair_potential``
    ``Phi_M(s)``            one half of the returned value
    ``W_M(s)``              returned scalar or NumPy array
    =====================  ================================================
    """
    return 4.0 * np.sum(
        pair_potential(_image_arguments(s, image_count), parameters), axis=-1)


def periodized_Wprime(s, parameters, image_count):
    """Return the derivative of the real-image approximation ``W_M``.

    Mathematical construction
    -------------------------
    Differentiating the finite image sum term by term gives

    ``W_M'(s) = 4*sum_{m=-M}^M V'(s - 2*pi*m)``.

    Notation-to-code map
    --------------------
    =====================  ================================================
    Mathematical notation  Code variable
    =====================  ================================================
    ``s``                   ``s``
    ``m``                   final-axis index made by ``_image_arguments``
    ``M``                   ``image_count``
    ``V'``                  ``pair_potential_prime``
    ``W_M'(s)``             returned scalar or NumPy array
    =====================  ================================================
    """
    return 4.0 * np.sum(
        pair_potential_prime(_image_arguments(s, image_count), parameters),
        axis=-1,
    )


def _transform_power_three(wavenumber, interlayer_distance):
    """Return the Fourier transform ``I_3(q)``.

    Mathematical construction
    -------------------------
    Using the transform convention

    ``I_3(q) = integral_R exp(-i*q*y)/(y^2 + L^2)^3 dy``

    and ``z = L*abs(q)``, the analytic value is

    ``I_3(q) = pi*exp(-z)*(z^2 + 3*z + 3)/(8*L^5)``.

    Notation-to-code map
    --------------------
    =====================  ===========================
    Mathematical notation  Code variable
    =====================  ===========================
    ``q``                   ``wavenumber``
    ``L``                   ``interlayer_distance``
    ``z=L*abs(q)``          ``z``
    ``z^2+3*z+3``           ``polynomial``
    ``I_3(q)``              returned value
    =====================  ===========================
    """
    z = interlayer_distance * np.abs(wavenumber)
    polynomial = z**2 + 3.0 * z + 3.0
    return (np.pi * np.exp(-z) * polynomial /
            (8.0 * interlayer_distance**5))


def _transform_power_six(wavenumber, interlayer_distance):
    """Return the Fourier transform ``I_6(q)``.

    Mathematical construction
    -------------------------
    Using the transform convention

    ``I_6(q) = integral_R exp(-i*q*y)/(y^2 + L^2)^6 dy``

    and ``z = L*abs(q)``, the analytic value is

    ``I_6(q) = pi*exp(-z)*P_5(z)/(3840*L^11)``,

    where

    ``P_5(z) = z^5 + 15*z^4 + 105*z^3 + 420*z^2 + 945*z + 945``.

    Notation-to-code map
    --------------------
    =====================  ===========================
    Mathematical notation  Code variable
    =====================  ===========================
    ``q``                   ``wavenumber``
    ``L``                   ``interlayer_distance``
    ``z=L*abs(q)``          ``z``
    ``P_5(z)``              ``polynomial``
    ``I_6(q)``              returned value
    =====================  ===========================
    """
    z = interlayer_distance * np.abs(wavenumber)
    polynomial = (z**5 + 15.0 * z**4 + 105.0 * z**3 +
                  420.0 * z**2 + 945.0 * z + 945.0)
    return (np.pi * np.exp(-z) * polynomial /
            (3840.0 * interlayer_distance**11))


def fourier_coefficients(parameters, max_mode):
    """Construct ``C_0`` and ``A_1,...,A_K`` for the Fourier cutoff.

    Mathematical construction
    -------------------------
    Poisson summation of the paper-matched potential gives

    ``W_K(s) = C_0 + sum_{k=1}^K A_k*cos(k*s)``.

    With ``q_k = 2*pi*k/a`` and the transforms ``I_3`` and ``I_6`` defined
    above,

    ``C_0 = (16*epsilon/a)*[sigma^12*I_6(0) - sigma^6*I_3(0)]``,

    ``A_k = (32*epsilon/a)*``
    ``[sigma^12*I_6(q_k) - sigma^6*I_3(q_k)]``, ``1 <= k <= K``.

    The Fourier cutoff is the upper mode ``K``. The constant mode ``C_0`` is
    retained separately and is not counted among the ``K`` cosine modes.

    Notation-to-code map
    --------------------
    =====================  ===========================================
    Mathematical notation  Code variable
    =====================  ===========================================
    ``epsilon``             ``epsilon`` from ``parameters.epsilon``
    ``sigma``               ``sigma`` from ``parameters.sigma``
    ``L``                   ``distance``
    ``a``                   ``lattice``
    ``K``                   ``max_mode``
    ``k=1,...,K``           entries of ``modes``
    ``q_k=2*pi*k/a``        entries of ``wavenumbers``
    ``I_3(q)``              ``_transform_power_three(q, distance)``
    ``I_6(q)``              ``_transform_power_six(q, distance)``
    ``C_0``                 returned ``constant``
    ``(A_1,...,A_K)``       returned array ``coefficients``
    =====================  ===========================================
    """
    if max_mode < 0:
        raise ValueError("max_mode must be nonnegative.")

    epsilon = parameters.epsilon
    sigma = parameters.sigma
    distance = parameters.interlayer_distance
    lattice = parameters.lattice_constant
    constant = (16.0 * epsilon / lattice *
                (sigma**12 * _transform_power_six(0.0, distance) -
                 sigma**6 * _transform_power_three(0.0, distance)))

    modes = np.arange(1, max_mode + 1, dtype=float)
    wavenumbers = TWOPI * modes / lattice
    coefficients = (32.0 * epsilon / lattice *
                    (sigma**12 * _transform_power_six(wavenumbers, distance) -
                     sigma**6 * _transform_power_three(wavenumbers, distance)))
    return float(constant), coefficients


def fourier_W(s, constant, cosine_coefficients):
    """Evaluate the Fourier-truncated periodic potential ``W_K(s)``.

    Mathematical construction
    -------------------------
    For ``K`` supplied cosine coefficients,

    ``W_K(s) = C_0 + sum_{k=1}^K A_k*cos(k*s)``.

    Here ``K`` is inferred from the length of ``cosine_coefficients`` rather
    than passed as a separate argument.

    Notation-to-code map
    --------------------
    =====================  ===========================================
    Mathematical notation  Code variable
    =====================  ===========================================
    ``s``                   ``s``; converted to the array ``phase``
    ``C_0``                 ``constant``
    ``(A_1,...,A_K)``       ``cosine_coefficients``; then ``coefficients``
    ``K``                   ``coefficients.size``
    ``k=1,...,K``           entries of ``modes``
    ``W_K(s)``              returned scalar or NumPy array
    =====================  ===========================================
    """
    phase = np.asarray(s, dtype=float)
    coefficients = np.asarray(cosine_coefficients, dtype=float)
    modes = np.arange(1, coefficients.size + 1, dtype=float)
    return constant + np.sum(
        coefficients * np.cos(phase[..., None] * modes), axis=-1)


def fourier_Wprime(s, cosine_coefficients):
    """Evaluate the derivative of the Fourier truncation ``W_K``.

    Mathematical construction
    -------------------------
    Differentiating the finite cosine series term by term gives

    ``W_K'(s) = -sum_{k=1}^K k*A_k*sin(k*s)``.

    As in :func:`fourier_W`, ``K`` is the number of supplied coefficients.

    Notation-to-code map
    --------------------
    =====================  ===========================================
    Mathematical notation  Code variable
    =====================  ===========================================
    ``s``                   ``s``; converted to the array ``phase``
    ``(A_1,...,A_K)``       ``cosine_coefficients``; then ``coefficients``
    ``K``                   ``coefficients.size``
    ``k=1,...,K``           entries of ``modes``
    ``W_K'(s)``             returned scalar or NumPy array
    =====================  ===========================================
    """
    phase = np.asarray(s, dtype=float)
    coefficients = np.asarray(cosine_coefficients, dtype=float)
    modes = np.arange(1, coefficients.size + 1, dtype=float)
    return -np.sum(
        modes * coefficients * np.sin(phase[..., None] * modes), axis=-1)
