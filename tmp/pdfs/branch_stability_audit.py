"""Read-only numerical audit for the deep-research branch analysis."""

from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import LinearOperator, eigsh

import experiments.relaxation.atomistic_two_chain_lj_lbfgs_convergence_study as model


OUTPUT = (Path(__file__).resolve().parents[2] / "outputs" / "relaxation" /
          "output_atomistic_two_chain_lj_lbfgs_convergence_study")


def theorem_path_lipschitz(start, end, N, cutoff=80, samples=401):
    """Sample the absolute-value Lipschitz bound in manuscript Lemma 6."""
    N1, N2, h1, h2, h = model.atomistic_geometry(N)
    images = np.arange(-cutoff, cutoff + 1)[None, :]
    largest_sum = 0.0
    for t in np.linspace(0.0, 1.0, samples):
        state = start + t * (end - start)
        u1, u2 = model.split_layers(state, N)
        for ui, opposite, Ni, No, hi, ho, sign in (
                (u1, u2, N1, N2, h1, h2, 1.0),
                (u2, u1, N2, N1, h2, h1, -1.0)):
            n = np.arange(Ni)[:, None]
            opposite_indices = (n + images) % No
            arguments = (sign * h * n + (h / ho) * ui[:, None] -
                         (h / hi) * opposite[opposite_indices] -
                         model.TWOPI * (h / hi) * images)
            largest_sum = max(
                largest_sum,
                float(np.max(np.sum(np.abs(model.Vsecond(arguments)), axis=1))))
    return 4.0 * np.sqrt(2.0) * largest_sum


def phase_supremum_lipschitz(N, cutoff=80, phase_samples=20001):
    """Approximate a state-independent phase supremum for Lemma 6's LF."""
    _, _, h1, h2, h = model.atomistic_geometry(N)
    largest_sum = 0.0
    for hi in (h1, h2):
        spacing = model.TWOPI * h / hi
        phase = np.linspace(-spacing / 2.0, spacing / 2.0,
                            phase_samples)[:, None]
        images = np.arange(-cutoff, cutoff + 1)[None, :]
        image_sum = np.sum(
            np.abs(model.Vsecond(phase - spacing * images)), axis=1)
        largest_sum = max(largest_sum, float(np.max(image_sum)))
    return 4.0 * np.sqrt(2.0) * largest_sum


def full_hessian(state, N, cutoff=80):
    """Assemble the exact finite-cutoff full Hessian as a sparse matrix."""
    N1, N2, h1, h2, h = model.atomistic_geometry(N)
    size = N1 + N2
    rows = []
    columns = []
    values = []

    for offset, Ni, hi in ((0, N1, h1), (N1, N2, h2)):
        for n in range(Ni):
            next_n = (n + 1) % Ni
            rows.extend((offset + n, offset + next_n,
                         offset + n, offset + next_n))
            columns.extend((offset + n, offset + next_n,
                            offset + next_n, offset + n))
            values.extend((1.0 / hi, 1.0 / hi, -1.0 / hi, -1.0 / hi))

    u1, u2 = model.split_layers(state, N)
    images = np.arange(-cutoff, cutoff + 1)
    for own_offset, opposite_offset, ui, opposite, Ni, No, hi, ho, sign in (
            (0, N1, u1, u2, N1, N2, h1, h2, 1.0),
            (N1, 0, u2, u1, N2, N1, h2, h1, -1.0)):
        alpha = h / ho
        beta = -h / hi
        for n in range(Ni):
            opposite_indices = (n + images) % No
            arguments = (sign * h * n + alpha * ui[n] -
                         (h / hi) * opposite[opposite_indices] -
                         model.TWOPI * (h / hi) * images)
            weights = hi * model.Vsecond(arguments)
            own_index = own_offset + n
            for opposite_index, weight in zip(opposite_indices, weights):
                other_index = opposite_offset + int(opposite_index)
                rows.extend((own_index, other_index, own_index, other_index))
                columns.extend((own_index, other_index,
                                other_index, own_index))
                values.extend((weight * alpha**2, weight * beta**2,
                               weight * alpha * beta,
                               weight * alpha * beta))
    return sparse.coo_matrix((values, (rows, columns)),
                             shape=(size, size)).tocsr()


def reduced_hessian_eigenvalues(state, N, cutoff=80):
    """Return four smallest eigenvalues in the optimizer's gauge coordinates."""
    hessian = full_hessian(state, N, cutoff)

    def product(vector):
        full_vector = model.build_mean_gauge(vector, N)
        return model.reduce_mean_gauge_gradient(hessian @ full_vector, N)

    operator = LinearOperator((2 * N, 2 * N), matvec=product, dtype=float)
    return np.sort(eigsh(operator, k=4, which="SA", tol=1.0e-10,
                         maxiter=10000, return_eigenvectors=False))


def hessian_vector_relative_error(state, N, cutoff=80):
    """Check the assembled Hessian against a centered gradient difference."""
    hessian = full_hessian(state, N, cutoff)
    coordinates = model.atomistic_reduced_coordinates(state, N)
    direction = np.random.default_rng(7401).normal(size=coordinates.size)
    direction /= np.linalg.norm(direction)
    exact = model.reduce_mean_gauge_gradient(
        hessian @ model.build_mean_gauge(direction, N), N)
    step = 1.0e-6
    plus = model.atomistic_value_gradient(
        coordinates + step * direction, N, cutoff)[1]
    minus = model.atomistic_value_gradient(
        coordinates - step * direction, N, cutoff)[1]
    difference = (plus - minus) / (2.0 * step)
    return np.linalg.norm(exact - difference) / max(1.0, np.linalg.norm(exact))


def continuum_translate(state, N, delta):
    """Apply the continuum translation orbit on the two native grids."""
    N1, N2, h1, h2, _ = model.atomistic_geometry(N)
    x1 = h1 * np.arange(N1)
    x2 = h2 * np.arange(N2)
    u1, u2 = model.split_layers(state, N)
    spline1 = CubicSpline(np.append(x1, model.TWOPI),
                          np.append(u1, u1[0]), bc_type="periodic")
    spline2 = CubicSpline(np.append(x2, model.TWOPI),
                          np.append(u2, u2[0]), bc_type="periodic")
    return np.concatenate((
        spline1(np.mod(x1 - delta, model.TWOPI)) - delta / 2.0,
        spline2(np.mod(x2 - delta, model.TWOPI)) + delta / 2.0,
    ))


def main():
    profiles = np.load(OUTPUT / "profiles.npz")
    multistart = np.load(OUTPUT / "multistart_finest_profiles.npz")
    N = int(multistart["N"])
    _, _, h1, _, h = model.atomistic_geometry(N)
    continuum = np.concatenate((profiles[f"u1_continuum_N{N}"],
                                profiles[f"u2_continuum_N{N}"]))
    sampled = np.concatenate((
        multistart["final_normalized_u1_sampled_continuum"],
        multistart["final_normalized_u2_sampled_continuum"]))
    seed3 = np.concatenate((
        multistart["final_normalized_u1_random_fourier_seed_3"],
        multistart["final_normalized_u2_random_fourier_seed_3"]))

    cp = h1**2 / (4.0 * np.sin(h1 / 2.0)**2)
    phase = np.linspace(-np.pi, np.pi, 20001)
    images = np.arange(-160, 161)
    wsecond = 4.0 * np.sum(
        model.Vsecond(phase[:, None] - model.TWOPI * images[None, :]), axis=1)

    print(f"N={N} h={h:.16e} CP={cp:.16e}")
    print(f"sampled_LF={theorem_path_lipschitz(continuum, sampled, N):.16e}")
    print(f"seed3_LF={theorem_path_lipschitz(continuum, seed3, N):.16e}")
    print(f"max_abs_Wsecond_M160={np.max(np.abs(wsecond)):.16e}")
    print(f"min_Wsecond_M160={np.min(wsecond):.16e}")
    print(f"max_Wsecond_M160={np.max(wsecond):.16e}")
    for trial_N in (2, 3, 4, 5, 10, 20, 40, 80, 160, 320, 640, 1280):
        _, _, trial_h1, _, _ = model.atomistic_geometry(trial_N)
        trial_cp = trial_h1**2 / (4.0 * np.sin(trial_h1 / 2.0)**2)
        trial_lf = phase_supremum_lipschitz(trial_N)
        print(f"phase_sup N={trial_N} CP={trial_cp:.12e} LF={trial_lf:.12e} "
              f"CP_LF={trial_cp * trial_lf:.12e}")
    sampled_error = model.two_layer_errors(sampled, continuum, N)
    seed3_direct = model.two_layer_errors(seed3, continuum, N)
    seed3_reflected = model.two_layer_errors(
        model.reflected_state(seed3, N), continuum, N)
    seed3_error = min(seed3_direct, seed3_reflected,
                      key=lambda item: item["sqrt_h2_error"])
    print("sampled_error", sampled_error)
    print("seed3_best_reflection_error", seed3_error)
    print(f"sampled_H2_over_h={sampled_error['sqrt_h2_error'] / h:.16e}")
    print(f"seed3_H2_over_h={seed3_error['sqrt_h2_error'] / h:.16e}")
    print("sampled_reduced_hessian_eigenvalues",
          reduced_hessian_eigenvalues(sampled, N))
    print("seed3_reduced_hessian_eigenvalues",
          reduced_hessian_eigenvalues(seed3, N))
    print("sampled_hessian_vector_relative_error",
          hessian_vector_relative_error(sampled, N))
    print("seed3_hessian_vector_relative_error",
          hessian_vector_relative_error(seed3, N))
    sampled1, sampled2 = model.split_layers(sampled, N)
    seed31, seed32 = model.split_layers(seed3, N)
    delta = -2.0 * (np.mean(seed31) - np.mean(sampled1))
    translated_sampled = continuum_translate(sampled, N, delta)
    aligned = model.two_layer_errors(seed3, translated_sampled, N)
    sampled_energy = model.raw_atomistic_value_gradient(
        sampled, N, model.PAIR_CUTOFF)[0]
    translated_energy = model.raw_atomistic_value_gradient(
        translated_sampled, N, model.PAIR_CUTOFF)[0]
    seed3_energy = model.raw_atomistic_value_gradient(
        seed3, N, model.PAIR_CUTOFF)[0]
    print(f"continuous_alignment_delta={delta:.16e}")
    print("continuous_alignment_error", aligned)
    print(f"continuous_alignment_energy_changes translated_minus_sampled="
          f"{translated_energy - sampled_energy:.16e} "
          f"seed3_minus_translated={seed3_energy - translated_energy:.16e}")

    print("seed3_refinement")
    continuum_spline = model.periodic_q_spline(
        profiles["continuum_x_N800"], profiles["continuum_q_N800"])
    for trial_N in (*model.N_VALUES, 640):
        if f"u1_continuum_N{trial_N}" in profiles:
            continuum_trial = np.concatenate((
                profiles[f"u1_continuum_N{trial_N}"],
                profiles[f"u2_continuum_N{trial_N}"]))
        else:
            continuum_trial = model.sample_continuum_pair(
                continuum_spline, trial_N)
        run = model.solve_atomistic_initial(
            model.random_fourier_atomistic_start(trial_N, 3),
            trial_N,
            model.PAIR_CUTOFF,
        )
        normalized_trial, normalization = model.appendix_b_normalize(
            run["u"], trial_N, model.PAIR_CUTOFF)
        direct = model.two_layer_errors(normalized_trial, continuum_trial, trial_N)
        reflected = model.two_layer_errors(
            model.reflected_state(normalized_trial, trial_N),
            continuum_trial,
            trial_N,
        )
        error = min(direct, reflected,
                    key=lambda item: item["sqrt_h2_error"])
        u1, u2 = model.split_layers(normalized_trial, trial_N)
        _, _, _, _, trial_h = model.atomistic_geometry(trial_N)
        mean_l2 = np.sqrt(model.TWOPI * (np.mean(u1)**2 + np.mean(u2)**2))
        print(
            f"N={trial_N} accepted={run['accepted']} "
            f"H2={error['sqrt_h2_error']:.12e} H2_over_h="
            f"{error['sqrt_h2_error'] / trial_h:.12e} "
            f"max={error['max_error']:.12e} means="
            f"({np.mean(u1):.12e},{np.mean(u2):.12e}) "
            f"mean_L2={mean_l2:.12e}"
        )


if __name__ == "__main__":
    main()
