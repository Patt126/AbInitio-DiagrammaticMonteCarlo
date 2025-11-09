# Diagrammatic Monte Carlo for Real Material Electron–Phonon Interactions

High-performance **Diagrammatic Monte Carlo (DiagMC)** framework for electron–phonon systems, developed in collaboration with **Cesare Franchini’s group (University of Vienna)** and aligned with ongoing developments around **VASP DFPT electron–phonon workflows**.

---

## Project Motivation

DiagMC stems from **quantum many-body field theory**, where electron–phonon interactions generate **emergent quasiparticles** such as *polarons*.  
Using the **Matsubara formalism**, the method stochastically evaluates the **finite-temperature Green’s functions and partition function** by sampling Feynman diagram topologies.

Key strengths:
- Non-perturbative treatment of interactions
- Direct evaluation of Matsubara Green’s functions
- No mean-field assumptions, systematic convergence

To model **real materials**, the solver must embed **first-principles electron–phonon coupling tensors** (DFT/DFPT from VASP).  
These tensors are large, high-dimensional, and must be queried **inside sampling loops executing thousands of diagram updates per second**.  
This makes **compression with controlled physical error** a core requirement.

---

## Code Architecture

| Component | Function |
|---|---|
| `diagmc/` | C++ solver for diagram sampling, topology updates, and Green’s function estimation |
| `python/` | Analysis utilities and numerical post-processing |

---

## Matrix Elements

Infrastructure for inspecting, compressing, and reconstructing **DFPT electron–phonon matrix elements** before ingestion in DiagMC sampling:

| File / Folder | Purpose |
|---|---|
| `matrix_element.ipynb` | Structural validation of `vaspelph.h5`, band structure/phonon plots, matrix-element amplitude maps, Bose–Einstein populations, Dulong-Petit limit check, band-gap renormalization diagnostics |
| `order_reduction.ipynb` | Data-driven compression following **Bernardi et al. PRL 125, 256402 (2020)**: SVD, symmetry extension, Wannier-basis reduction (Wannier90), k-space reconstruction, error analysis via physical observables |
| `DATA/` | Minimal coarse-mesh tensors provided for local execution and testing |
| `RESULTS/` | Benchmarks on production-scale k-meshes, compression trends, reconstruction error, physical observable drift |

**Note:** Effective compression requires **fine k-mesh tensors**.  
Coarse data in `DATA/` is for execution tests only; full compression analysis refers to high-resolution results in `RESULTS/`.

---

## Physics Scope

- Emergent quasiparticles (polaronic dressing)
- Electron–phonon self-energy at finite temperature
- Band-gap renormalization from interacting propagators
- Scalable integration of ab-initio couplings into Monte-Carlo estimators

---

## Constraints Driving Design

- Sampling loops evaluate **10⁶–10⁸ diagrams per run**
- e–ph tensors must be **compressed, differentiable, and reconstructible**
- Memory and lookup overhead must not compete with diagram sampling time
- Error must be controlled in **physical observables**, not just matrix norms

---

This codebase links **DFT-level material specificity** with **many-body stochastic solvers**, under the performance constraints imposed by real DiagMC workloads.
