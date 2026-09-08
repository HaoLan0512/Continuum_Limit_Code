# Continuum Limit Relaxation Code

This repository contains an auditable one-dimensional continuum-limit workflow:

    Lennard-Jones pair potentials
        -> periodic stacking potential W
        -> relaxed displacement or disregistry
        -> Bloch-band calculations

The organized source tree separates reusable Bloch solvers in `clr` from runnable research experiments in `experiments`. Exploratory scripts and unchanged notebooks are under `exploratory`; an exact solver duplicate is retained under `archive` for provenance. Generated research artifacts are grouped by their verified producer under `outputs`.

The principal method families are:

- Lennard-Jones 12-6 lattice sums, registry mappings, parameter sweeps, and convention checks;
- finite-difference L-BFGS-B relaxation, J[v] variants, and finite-difference symbols applied through FFT;
- fourth-order finite-difference Bloch matrices and plane-wave convolution with finite-difference kinetic symbols.

See [CODE_HIERARCHY.md](CODE_HIERARCHY.md) for the complete file inventory, dependency relationships, old-to-new path mapping, output provenance, run commands, and validation record.
