# CLR Code Hierarchy

This document is the authoritative map of the organized one-dimensional continuum-limit relaxation research code. The reorganization is structural: numerical algorithms, equations, grids, parameters, solver settings, conventions, and public computational function signatures were preserved.

## Workflow taxonomy

1. Potentials: Lennard-Jones 12-6 lattice sums, phase/registry maps, parameter sweeps, diagnostics, and convention comparison.
2. Relaxation: finite-difference L-BFGS-B, J[v] variants, and finite-difference derivative or Laplacian symbols evaluated through FFT.
3. Band structure: fourth-order finite-difference Bloch matrices and plane-wave potential convolution with second- or fourth-order finite-difference kinetic symbols.
4. Exploratory material: historical notebooks and a circular-distance sanity script.
5. Archive: an exact source duplicate and generated artifacts whose active producer is unresolved.

Names containing fd_symbol_fft deliberately distinguish FFT application of finite-difference symbols from a fully spectral discretization.

## Organized tree

    Continuum_Limit_Code/
    +-- CODE_SUMMARY.md
    +-- CODE_HIERARCHY.md
    +-- clr/
    |   +-- __init__.py
    |   +-- potentials/
    |   |   +-- __init__.py
    |   |   +-- lj_periodic.py
    |   +-- band_structure/
    |   |   +-- __init__.py
    |   |   +-- bloch_finite_difference.py
    |   |   +-- bloch_fd_symbol_fft.py
    |   +-- relaxation/
    |       +-- __init__.py
    |       +-- odd_periodic.py
    +-- experiments/
    |   +-- __init__.py
    |   +-- potentials/
    |   |   +-- __init__.py
    |   |   +-- lj_stacking_potential_exploration.py
    |   |   +-- lj_lattice_sum_parameter_sweep.py
    |   |   +-- lj_registry_diagnostic_parameter_sweep.py
    |   |   +-- lj_potential_convention_comparison.py
    |   +-- relaxation/
    |   |   +-- __init__.py
    |   |   +-- finite_difference_lbfgs_single_case.py
    |   |   +-- jv_finite_difference_lbfgs_single_case.py
    |   |   +-- jv_phase_fixed_finite_difference_lbfgs_single_case.py
    |   |   +-- jv_unrestricted_finite_difference_lbfgs_mesh_study.py
    |   |   +-- jv_unrestricted_lj_fourier_finite_difference_lbfgs_mesh_study.py
    |   |   +-- atomistic_two_chain_lj_lbfgs_convergence_study.py
    |   |   +-- atomistic_two_chain_lj_fourier_continuum_convergence_comparison.py
    |   |   +-- ATOMISTIC_MINIMIZATION_CODE_MAP.md
    |   |   +-- fd_symbol_fft_virtual_time_single_case.py
    |   |   +-- fd_symbol_fft_constraint_comparison.py
    |   |   +-- fd_symbol_fft_diagnostics_comparison.py
    |   |   +-- fd_symbol_fft_lbfgs_single_case.py
    |   +-- band_structure/
    |       +-- __init__.py
    |       +-- bloch_finite_difference_band_study.py
    |       +-- bloch_fd_symbol_fft_band_study.py
    +-- exploratory/
    |   +-- scripts/circular_distance_sanity_check.py
    |   +-- notebooks/
    |       +-- fft_picard_relaxation.ipynb
    |       +-- fd_symbol_fft_relaxation_prototype.ipynb
    |       +-- jfnk_relaxation.ipynb
    |       +-- split_radix_fft_continuation.ipynb
    |       +-- discrete_gsfe_lbfgsb_relaxation.ipynb
    |       +-- bloch_solver_comparison.ipynb
    +-- archive/duplicates/bloch_finite_difference_duplicate.py
    +-- outputs/
        +-- potentials/
        +-- relaxation/
        +-- band_structure/
        +-- archive/

Existing repository configuration, AGENTS.md, .git, .vscode, and ignored caches remain outside the reorganization.

## Python inventory

| Path | Method | Role | Principal inputs | Outputs and location | Local dependencies |
|---|---|---|---|---|---|
| clr/__init__.py | None | Package marker | None | None | None |
| clr/potentials/__init__.py | None | Potential-module namespace marker | None | None | lj_periodic |
| clr/potentials/lj_periodic.py | Effective LJ pair interaction; real-image periodization; analytic Poisson/Fourier coefficients | Reusable paper-matched V, W, W', and Fourier-cutoff evaluator | epsilon, sigma, fixed interlayer distance, lattice constant, phase, cutoff | arrays and coefficients returned to caller | None |
| clr/band_structure/__init__.py | None | Bloch-solver namespace marker | None | None | None |
| clr/band_structure/bloch_finite_difference.py | Fourth-order periodic finite differences | Reusable Bloch eigensolver | grid, periodic potential, momentum, number of bands | arrays returned to caller | None |
| clr/band_structure/bloch_fd_symbol_fft.py | Plane-wave convolution; selectable FD kinetic symbol | Reusable Bloch eigensolver and guarded comparison | grid, periodic potential, momentum, bands, FD order | arrays; optional interactive comparison | bloch_finite_difference for guarded comparison |
| clr/relaxation/__init__.py | None | Relaxation-helper namespace marker | None | None | None |
| clr/relaxation/odd_periodic.py | Odd periodic reconstruction; transpose-Jacobian reduction | Reusable odd, phase-fixed optimization coordinates | reduced values, optional full grid size, or full gradient | arrays returned to caller | None |
| experiments/__init__.py | None | Experiment namespace marker | None | None | None |
| experiments/potentials/__init__.py | None | Potential experiment namespace marker | None | None | None |
| experiments/potentials/lj_stacking_potential_exploration.py | LJ lattice sum in registry space | Interactive exploration | sigma, spacing, registry, truncation | displayed figures only | None |
| experiments/potentials/lj_lattice_sum_parameter_sweep.py | Basis-aware LJ lattice sum | CLI profile and sigma-spacing sweep | CLI options and monoatomic/diatomic basis | six artifacts under outputs/potentials/output_lj_lattice_sum_parameter_sweep by default | None |
| experiments/potentials/lj_registry_diagnostic_parameter_sweep.py | Monatomic LJ lattice sum with registry diagnostics | CLI sweep and consistency diagnostics | CLI options, sigma, spacing, registry grid | six artifacts under outputs/potentials/output_lj_registry_diagnostic_parameter_sweep by default | None |
| experiments/potentials/lj_potential_convention_comparison.py | Independent LJ convention implementations | Scaling, derivative, and AA/AB comparison | fixed comparison parameters | comparison_report.txt under outputs/potentials/output_lj_potential_convention_comparison | None |
| experiments/relaxation/__init__.py | None | Relaxation experiment namespace marker | None | None | None |
| experiments/relaxation/finite_difference_lbfgs_single_case.py | Centered finite differences; L-BFGS-B | 200-point continuum-energy minimizer | coded parameters N=200 and epsilon=0.1 | u_min.npy and plots under outputs/relaxation/output_finite_difference_lbfgs_single_case | None |
| experiments/relaxation/jv_finite_difference_lbfgs_single_case.py | Forward finite differences; L-BFGS-B | J[v] experiment through v=x+u | coded 400-point grid and energy parameters | u_min.npy, v_min.npy, and plots under outputs/relaxation/output_jv_finite_difference_lbfgs_single_case | None |
| experiments/relaxation/jv_phase_fixed_finite_difference_lbfgs_single_case.py | Forward differences; odd parametrization; L-BFGS-B | Phase-fixed J[v] minimizer and constraints | coded grid and energy parameters | arrays and plots under outputs/relaxation/output_jv_phase_fixed_finite_difference_lbfgs_single_case | clr.relaxation.odd_periodic |
| experiments/relaxation/jv_unrestricted_finite_difference_lbfgs_mesh_study.py | Forward differences; unrestricted and odd-reference L-BFGS-B; mesh refinement | Independent numerical check of Theorem 1 through nonsymmetric multistart and postprocessed phase alignment | N=100, 200, 400; alpha=1; W=2*cos; deterministic Fourier starts | CSV, NPZ, report, and plots under outputs/relaxation/output_jv_unrestricted_finite_difference_lbfgs_mesh_study | clr.relaxation.odd_periodic |
| experiments/relaxation/jv_unrestricted_lj_fourier_finite_difference_lbfgs_mesh_study.py | Analytic LJ Fourier cutoff; forward differences; unrestricted and odd-reference L-BFGS-B | Repeat the unrestricted mesh study with the pair-derived K=5 continuum potential | N=100, 200, 400; a=1, sigma=0.9, L=1, epsilon=0.5 | CSV, NPZ, report, and plots under outputs/relaxation/output_jv_unrestricted_lj_fourier_finite_difference_lbfgs_mesh_study | clr.potentials.lj_periodic; cosine mesh runner |
| experiments/relaxation/atomistic_two_chain_lj_lbfgs_convergence_study.py | Two-chain long-range LJ energy; exact truncated gradient; phase-fixed continuum finite differences; L-BFGS-B | Compare two fixed atomistic solution tracks for N=20, 40, 80, 160, 320, 640 with the matched continuum minimizer in the paper's discrete norms | N and N+1 periodic chains, finite LJ image sum, sampled-continuum warm start, fixed random-Fourier seed 3, and 400- and 800-point deterministic continuum references | convergence_summary.csv, profiles.npz, convergence_loglog.png, and check_report.txt under outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study | clr.potentials.lj_periodic; clr.relaxation.odd_periodic |
| experiments/relaxation/atomistic_two_chain_lj_fourier_continuum_convergence_comparison.py | Analytic LJ Fourier continuum; finite-difference L-BFGS-B; saved-profile postprocessing | Recompute atomistic convergence against K=5 and compare K=6 and M=160 continuum references | validated atomistic profiles.npz, a=1, sigma=0.9, L=1, epsilon=0.5 | CSV, NPZ, report, and plot under outputs/relaxation/output_atomistic_two_chain_lj_fourier_continuum_convergence_comparison | clr.potentials.lj_periodic; atomistic convergence helpers |
| experiments/relaxation/fd_symbol_fft_virtual_time_single_case.py | FD Laplacian symbols applied through FFT | Picard and semi-implicit virtual-time solvers | grid, epsilon, iterations, tolerance | arrays, console diagnostics, displayed figure | None |
| experiments/relaxation/fd_symbol_fft_constraint_comparison.py | FD symbols through FFT; projected virtual time | Constraint and initial-guess comparison | solver options and admissible-set projections | arrays, console diagnostics, displayed figure | None |
| experiments/relaxation/fd_symbol_fft_diagnostics_comparison.py | FD derivative symbols through FFT; projected virtual time | Constraint, residual, and energy-history comparison | solver options, pinning, initial guesses | histories, console diagnostics, displayed figures | None |
| experiments/relaxation/fd_symbol_fft_lbfgs_single_case.py | Centered FD derivative symbols applied through FFT; L-BFGS-B | 256-point single-case minimizer | coded grid and energy parameters | u_min_fft.npy and plots under outputs/relaxation/output_fd_symbol_fft_lbfgs_single_case | None |
| experiments/band_structure/__init__.py | None | Band-study namespace marker | None | None | None |
| experiments/band_structure/bloch_finite_difference_band_study.py | Fourth-order finite-difference Bloch solver | Momentum sweep, four bands, modes, and displacement plots | 200-point relaxation output | nine PDFs under outputs/band_structure/output_bloch_finite_difference_band_study | clr.band_structure.bloch_finite_difference |
| experiments/band_structure/bloch_fd_symbol_fft_band_study.py | Plane-wave convolution with fourth-order FD kinetic symbol | Momentum sweep, four bands, modes, and displacement plots | 256-point FFT-symbol relaxation output | nine PDFs under outputs/band_structure/output_bloch_fd_symbol_fft_band_study | clr.band_structure.bloch_fd_symbol_fft |
| exploratory/scripts/circular_distance_sanity_check.py | Periodic circular distance | Console sanity comparison | coded phase values | console only | None |
| archive/duplicates/bloch_finite_difference_duplicate.py | Fourth-order periodic finite differences | Preserved exact pre-header duplicate | grid, potential, momentum, bands | arrays if called; no active consumer | None |

## Verified relationships

- The archived finite-difference Bloch file was byte-for-byte identical to the former root bloch.py before documentation headers were added. Active code never imports the archived copy.
- The two band drivers remain distinct: one uses a 200-point finite-difference matrix solver and the other uses a 256-point plane-wave representation with a finite-difference kinetic symbol.
- The general and diagnostic Lennard-Jones generators overlap but retain different defaults, basis support, and diagnostic behavior. They were not merged.
- The convention-comparison script intentionally contains independent implementations rather than importing either producer.
- The reusable LJ module implements the paper's factor W=4*sum_m V. The older potential-sweep default prefactor 2 represents Phi instead.
- The reusable odd-periodic module is imported directly by the phase-fixed single case, the cosine unrestricted mesh-study core, and the atomistic study's continuum solver. The LJ-Fourier mesh study and Fourier continuum comparison use it transitively through those two latter workflows.
- The cosine unrestricted study retains its original W=2*cos entry point. The LJ-Fourier study imports its numerical workflow but writes to a separate producer-owned output directory.
- The Fourier-continuum comparison reuses the warm-track profiles through the
  atomistic producer's unsuffixed NPZ aliases because changing only the
  continuum evaluator does not change the atomistic objective; K=6 and M=160
  references quantify the resulting approximation error.
- The atomistic convergence study reports `sampled_continuum` and
  `random_fourier_seed_3` as two independent tracks; it does not select between
  them. An endpoint is accepted when it is finite, L-BFGS-B reports success,
  Appendix-B normalization holds, and
  `el_residual_inf_norm / h <= 1e-2`. Seed 3 is an intentionally selected
  phase-stress realization whose finite-grid sawtooth is assessed through an
  empirical O(h) envelope; it is not representative random-start statistics
  or evidence of a global minimum.
- The constraint and diagnostics virtual-time scripts are related extensions of the base solver family but have no import dependency on it.
- The finite-difference band study reads only the 200-point finite-difference relaxation result. The FFT-symbol study reads only the 256-point FFT-applied finite-difference-symbol result.
- The archived u_original.png and v_original.png have no verified producer in the committed source.

## Output mapping

| Former location | Organized location | Producer or status |
|---|---|---|
| root minimizer_eps_*.pdf and modified_disregistry_eps_0.1.pdf | outputs/relaxation/output_finite_difference_lbfgs_single_case | finite_difference_lbfgs_single_case.py |
| regenerated 200-point u_min.npy | outputs/relaxation/output_finite_difference_lbfgs_single_case | unchanged finite-difference L-BFGS-B calculation |
| root u_min.npy, v_min.npy, minimizer_u.pdf, minimizer_v.pdf | outputs/relaxation/output_jv_finite_difference_lbfgs_single_case | jv_finite_difference_lbfgs_single_case.py |
| output phase-fixed arrays and resolved images | outputs/relaxation/output_jv_phase_fixed_finite_difference_lbfgs_single_case | jv_phase_fixed_finite_difference_lbfgs_single_case.py |
| None (new numerical study) | outputs/relaxation/output_jv_unrestricted_finite_difference_lbfgs_mesh_study | jv_unrestricted_finite_difference_lbfgs_mesh_study.py |
| None (new numerical study) | outputs/relaxation/output_jv_unrestricted_lj_fourier_finite_difference_lbfgs_mesh_study | jv_unrestricted_lj_fourier_finite_difference_lbfgs_mesh_study.py |
| Three-start atomistic study snapshot | outputs/archive/output_atomistic_two_chain_lj_lbfgs_convergence_study_three_start_snapshot_20260906.zip | Preserved baseline with SHA-256 inventory |
| Seven-start atomistic study snapshot | outputs/archive/output_atomistic_two_chain_lj_lbfgs_convergence_study_seven_start_snapshot_20260907.zip | Preserved baseline with SHA-256 inventory |
| None (new numerical study) | outputs/relaxation/output_atomistic_two_chain_lj_lbfgs_convergence_study | atomistic_two_chain_lj_lbfgs_convergence_study.py; four live two-track outputs |
| Validated atomistic profiles | outputs/relaxation/output_atomistic_two_chain_lj_fourier_continuum_convergence_comparison | atomistic_two_chain_lj_fourier_continuum_convergence_comparison.py |
| FFT base relaxation artifacts | outputs/relaxation/output_fd_symbol_fft_lbfgs_single_case | fd_symbol_fft_lbfgs_single_case.py |
| output_pair_potential | outputs/potentials/output_lj_lattice_sum_parameter_sweep | lj_lattice_sum_parameter_sweep.py |
| output_pair_potential_test | outputs/potentials/output_lj_registry_diagnostic_parameter_sweep | lj_registry_diagnostic_parameter_sweep.py |
| output_code_comparison | outputs/potentials/output_lj_potential_convention_comparison | lj_potential_convention_comparison.py |
| results_fd | outputs/band_structure/output_bloch_finite_difference_band_study | bloch_finite_difference_band_study.py |
| result_fft | outputs/band_structure/output_bloch_fd_symbol_fft_band_study | bloch_fd_symbol_fft_band_study.py |
| output_pair_potential_test.zip | outputs/archive/output_lj_registry_diagnostic_parameter_sweep_snapshot.zip | preserved exact snapshot |
| output/u_original.png and output/v_original.png | outputs/archive/unresolved_phase_fixed_outputs | producer unresolved |

Existing artifacts were moved without regeneration. The separate 200-point finite-difference u_min.npy was regenerated from unchanged source because the historical matching band input was absent.

## Source path migration

| Former path | Organized path |
|---|---|
| plot_lj_stacking_potential.py | experiments/potentials/lj_stacking_potential_exploration.py |
| build_W_from_pair_potential.py | experiments/potentials/lj_lattice_sum_parameter_sweep.py |
| build_W_from_LJ.py | experiments/potentials/lj_registry_diagnostic_parameter_sweep.py |
| compare_lj_potential_implementations.py | experiments/potentials/lj_potential_convention_comparison.py |
| relaxation.py | experiments/relaxation/finite_difference_lbfgs_single_case.py |
| relaxation _Jv.py | experiments/relaxation/jv_finite_difference_lbfgs_single_case.py |
| relaxation_Jv_phase_fixed.py | experiments/relaxation/jv_phase_fixed_finite_difference_lbfgs_single_case.py |
| 1D_fd_fft.py | experiments/relaxation/fd_symbol_fft_virtual_time_single_case.py |
| 1D_fd_fft_custom.py | experiments/relaxation/fd_symbol_fft_constraint_comparison.py |
| 1D_fd_fft_custom_v1.py | experiments/relaxation/fd_symbol_fft_diagnostics_comparison.py |
| FFT base/relaxation_fft.py | experiments/relaxation/fd_symbol_fft_lbfgs_single_case.py |
| bloch.py | clr/band_structure/bloch_finite_difference.py |
| FFT base/bloch.py | archive/duplicates/bloch_finite_difference_duplicate.py |
| FFT base/bloch_fft.py | clr/band_structure/bloch_fd_symbol_fft.py |
| main_bands.py | experiments/band_structure/bloch_finite_difference_band_study.py |
| FFT base/main_bands_fft.py | experiments/band_structure/bloch_fd_symbol_fft_band_study.py |
| test.py | exploratory/scripts/circular_distance_sanity_check.py |

## Notebook migration

| Former path | Organized path |
|---|---|
| 1D_FFT.ipynb | exploratory/notebooks/fft_picard_relaxation.ipynb |
| 1D_fd_fft.ipynb | exploratory/notebooks/fd_symbol_fft_relaxation_prototype.ipynb |
| 1D_JFNK.ipynb | exploratory/notebooks/jfnk_relaxation.ipynb |
| 1D_split_radix_FFT.ipynb | exploratory/notebooks/split_radix_fft_continuation.ipynb |
| one-dimensional moire materials.ipynb | exploratory/notebooks/discrete_gsfe_lbfgsb_relaxation.ipynb |
| FFT base/test.ipynb | exploratory/notebooks/bloch_solver_comparison.ipynb |

Notebook contents and saved cell outputs were not edited or re-executed. The JFNK notebook remains historical exploratory evidence and contains a saved KeyboardInterrupt.

## Representative entry points

Run modules from the repository root with the selected relax Conda interpreter:

    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.potentials.lj_lattice_sum_parameter_sweep
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.potentials.lj_registry_diagnostic_parameter_sweep
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.potentials.lj_potential_convention_comparison
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.relaxation.finite_difference_lbfgs_single_case
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.relaxation.jv_finite_difference_lbfgs_single_case
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.relaxation.jv_phase_fixed_finite_difference_lbfgs_single_case
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.relaxation.jv_unrestricted_finite_difference_lbfgs_mesh_study
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.relaxation.jv_unrestricted_lj_fourier_finite_difference_lbfgs_mesh_study
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.relaxation.atomistic_two_chain_lj_lbfgs_convergence_study
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.relaxation.atomistic_two_chain_lj_fourier_continuum_convergence_comparison
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.relaxation.fd_symbol_fft_lbfgs_single_case
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.band_structure.bloch_finite_difference_band_study
    C:\Users\lh201\anaconda3\envs\relax\python.exe -m experiments.band_structure.bloch_fd_symbol_fft_band_study

The two default LJ CLI output directories are repository-relative. An explicitly supplied relative --output-dir remains relative to the caller's working directory.

## Validation record

Environment selected for validation:

- Python 3.12.12 from the relax Conda environment
- NumPy 2.4.2
- SciPy 1.17.1
- Matplotlib 3.10.8

Preflight recorded the clean tracked baseline at commit 5de6374, hashes for 17 Python sources and six notebooks, and a manifest for 49 existing generated artifacts totaling 4,079,282 bytes. A byte-for-byte rollback copy of those artifacts was retained outside the repository during validation.

Final validation results:

- Syntax and imports: all 23 Python files compiled with an external bytecode
  cache, and both packaged solvers plus every import-safe experiment imported.
- Numerical comparison: Lennard-Jones values and derivatives, reduced
  virtual-time cases, the 64-point FFT/FD-symbol minimizer, both 400-point J[v]
  solvers, and 32-point Bloch cases matched their pre-move baselines at the
  approved rtol=1e-12 and atol=1e-12. Most results were bitwise identical; the
  largest absolute difference was 2.771e-13 in a recorded FFT residual.
  Eigenvector phase-invariant overlap errors were at most 2.220e-16.
- Default finite-difference producer: the unchanged N=200 run converged in an
  isolated tree and reproduced the retained u_min.npy bitwise.
- Preservation: all 49 pre-existing artifacts retain their SHA-256 hashes and
  original total size of 4,079,282 bytes. All six notebooks retain their hashes.
- Inventories: output-directory counts, 1,681-row and 14,641-row CSVs, six ZIP
  entry hashes, NPY shapes and dtypes, J[v] identity, and phase constraints pass.
- Producer smoke tests: reduced five-by-five potential sweeps created all six
  expected files each; the convention report hash matched. Reduced band drivers
  preserved band arrays and created all nine files only in their new directories.
- Band-driver warning: both original and organized reduced drivers emit the same
  eight NumPy ComplexWarning instances while Matplotlib implicitly casts complex
  perturbation curves to real values. This is pre-existing plotting behavior,
  was not changed by the approved structural scope, and prevents a warning-clean
  acceptance result.
- Stale paths and diff hygiene: operational searches found no legacy imports,
  working-directory changes, obsolete output locations, or dead absolute path.
  Former names remain only in migration documentation. git diff --check passes;
  Git reports only line-ending conversion notices for the Windows worktree.

The expensive full 121 by 121 diagnostic sweep, default 512-point virtual-time runs, full band drivers, and notebook re-execution remain intentionally unrun and require separate approval.

### 2026-09-01 analytic LJ Fourier addition

- Added the reusable effective LJ pair potential, its first two derivatives,
  the paper-matched real-image periodization `W=4*sum_m V`, and closed analytic
  Poisson/Fourier coefficients using NumPy only.
- Pair values and first derivatives match the former atomistic implementation
  exactly on the regression grid; the largest second-derivative difference is
  `8.327e-17`. Full and reduced atomistic directional-gradient errors remain
  `8.940e-10` and `3.009e-10`.
- K=5 versus K=6 changes `W` by `1.7454e-11` and `W'` by `1.0472e-10`.
  The coefficient bound for `Lip(W')` is `0.2304661 < 1`, and the checked
  evenness, oddness, monotonicity, and `W'/s` conditions pass.
- The cosine unrestricted study was rerun in memory after callable
  parameterization. Its N=100, 200, 400 best energies agree with the saved
  baseline to at most `8.882e-16`, and all original checks pass. Its saved
  output hashes are unchanged.
- The new LJ-Fourier unrestricted N=100, 200, 400 study passes all optimizer,
  Euler-Lagrange, monotonicity, alignment, multistart, mesh-refinement, and
  cutoff checks. The LJ-specific accepted thresholds are reduced-gradient
  `3e-7` and Euler-Lagrange residual `2e-5`; the finest observed values are
  `2.690e-7` and `1.713e-5`.
- Reusing the validated atomistic profiles, the K=5 continuum comparison gives
  log-log slopes `-1.000007` in the paper-consistent discrete H2 norm and
  `-1.057039` in the maximum norm. K=6, M=160, and 400/800
  continuum-reference changes are each below 5 percent of the finest
  atomistic error.
- The original cosine and atomistic output directories retain every recorded
  SHA-256 hash. The two new producers created six and four artifacts in their
  own output directories. Their three PNG figures were rendered and visually
  inspected without layout defects.
- `python -m compileall -q .`, targeted imports, and `git diff --check` pass
  under the relax environment. The expensive atomistic minimization itself was
  not rerun because its pair model, parameters, and saved normalized solutions
  are unchanged.

### 2026-09-05 reusable odd-variable reduction

- Added `clr/relaxation/odd_periodic.py` as the single implementation of the
  odd reconstruction `[0,a,0,-a[::-1]]` and its transpose-Jacobian gradient
  reduction. The three former experiment-local implementations are now import
  aliases, so their established callable names remain available.
- Reconstruction and reduced-gradient arrays are bitwise identical to the
  pre-extraction implementations at 100, 200, 400, and 800 grid points. The
  fixed-node and discrete-oddness defects are exactly zero.
- Centered directional-gradient relative errors are `5.50e-10` for the
  phase-fixed single case, `7.85e-11` for the cosine mesh objective, and
  `3.39e-10` for the real-image LJ continuum objective.
- Pre/post numerical fingerprints match exactly for the phase-fixed solve,
  the complete cosine and LJ-Fourier mesh studies, the 400/800 real-image
  continuum references, and the K=5, K=6, and M=160 Fourier-comparison
  references. Every existing acceptance check passes.
- All 25 retained artifacts in the five affected output directories preserve
  their SHA-256 hashes. Targeted imports, alias-identity checks, `compileall`,
  stale-definition searches, trailing-whitespace checks, and `git diff
  --check` pass under Python 3.12.12 in the relax environment.
- The full atomistic N=20,...,320 optimization was not rerun because only its
  separately validated continuum-reference coordinate map changed location.

### 2026-09-07 two-track atomistic convergence study

- Archived the former seven-start outputs, with inventory and SHA-256 hashes,
  as `outputs/archive/output_atomistic_two_chain_lj_lbfgs_convergence_study_seven_start_snapshot_20260907.zip`.
- The full two-track run accepted all 12 endpoints and passed all 23 focused
  checks. The warm-start H2 and maximum slopes are `-0.996919` and `-0.999255`;
  the seed-3 slopes are `-1.080796` and `-1.081950`.
- The seed-3 H2 plateau breaks with
  `error(640)/error(320)=0.199647`, while the largest observed H2/h ratio is
  `0.802343`. These values support an empirical O(h) envelope, not smooth
  pointwise halving.
- For N through 320, retained energies differ from the archived run by at most
  `8.882e-14`; warm errors and saved profiles, and the saved N=320 seed-3
  profile, are bitwise identical.
- The 12-row CSV and 70-key NPZ schemas, warm-profile aliases, downstream
  loader, plot rendering, scoped `compileall`, stale-label searches, and
  `git diff --check` pass. The live output directory contains only the four
  documented artifacts.
