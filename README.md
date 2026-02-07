# Diagrammatic Monte Carlo for Ab-Initio Electron–Phonon Physics

Framework connecting **Diagrammatic Monte Carlo (DiagMC)** to **first-principles electron–phonon matrix elements**.  
Developed in collaboration with **Cesare Franchini’s group (University of Vienna)**.

The goal is to bring realistic material couplings into **finite-temperature many-body Feynman diagram sampling**, under the constraint of **10⁶–10⁸ diagram evaluations per run**, requiring structured tensor compression and low-latency reconstruction. 

---

## Diagrammatic Monte Carlo

### Physical backbone

In the Matsubara formalism, the interacting Green’s function is written as a diagrammatic expansion over all connected Feynman topologies \(D\):

```math
G(k, \tau) = \sum_{D} \int \mathcal{D}[\tau_i, q_i] \;
\mathcal{W}_D(k,\{\tau_i, q_i\}) ,
```

where each diagram contributes a weight \(W\) built from electron and phonon propagators and electron–phonon vertices.  
DiagMC stochastically samples diagram space by proposing topology updates, evaluating weights, and accumulating Green’s function estimators at finite temperature.

This formulation captures **emergent quasiparticles (e.g. polarons)** without mean-field or low-order truncations, but demands extremely fast access to electron–phonon coupling data inside the Monte-Carlo kernel.

### Contents

## DiagMC_Holstein
Basic diagrammatic montecarlo sampling for the Holstein-Model: an important prototype Hamiltonian were the coupling is just a costat (g), in good agreement with literature implementations.

## DiagMC_momentum
-Extension of Holstein Code to include electron dispersion, phonon frequency and coupling which explicitly depends on particles momentum and mode/band indices. This implies an improvment of the procedure needed to compute diagram weight efficiently: now an update propagate across the whole data structure, modifiyng the structure of vertices and propagators. 
- Benchmark over the Breathing-Mode model showing good agreement with the theoretical perturbation calculation (ref: Momentum average approximation for models with electron-phonon coupling dependent on the phonon momentum| G.L.Goodvin & M.Berciu)
- Benchmark of the sign-problem: instability caused by phase oscillating coupling.

Common structure:
| File | Description |
|---|---|
| `DMC.cpp` | Monte-Carlo sampling loop, diagram updates, estimators |
| `feynmanDiagram.{cpp,h}` | Linked-list diagram representation and topology operations |
| `saveSimulationToFile.{cpp,h}` | Observables, distributions, configuration dumps |
| `analyze.py` | Post-processing of \(G(\tau)\), energies, diagram statistics |


---
## Matrix_Elements_QE

Tools for **validating, compressing, and reconstructing** DFT electron–phonon tensors from QE +EPW, effieciently on fine grid momentum and band indices before and during integration into DiagMC.

Compression is necessary because:
- tensors scale with:
```math
   O(N_k \times N_q \times N_b^2)
```
- the DiagMC kernel performs **millions of weight evaluations per second**
- reconstruction must preserve **physical observables**, not just matrix norms
- The code implement SVD or Tucker decomposition that remove the main bottleneck in performing the  Wannier-Fourier interpolation. This is achieved because this low-rank factorization techniques allow to factorize the densor over different dimension and process them indipendently.
---

## Matrix Elements (VASP)

Initial Implementation in VASP, showing some limitation on the possibilities to achieve an efficient procedure. Likely new feature will be soon available on the code to reproduce the result achieved with QE+EPW.

### Contents

| File / Folder | Description |
|---|---|
| `matrix_element.ipynb` | Reads `vaspelph.h5`, inspects tensor structure, plots electronic bands, phonons, vertex magnitude, Bose–Einstein occupation, Dulong–Petit limit, band-gap renormalization diagnostics |
| `order_reduction.ipynb` | Implements **SVD and Wannier-space compression** following *BY. Luo, D. C. Desai, B. K. Chang, C.-H. Park, and M. Bernardi, Data-Driven Compression of Electron-Phonon Interactions, Phys. Rev. X 14, 021023 (2024))*. Includes symmetry extension, singular-value trend analysis, Wannier90 rotation, k-space reconstruction, and comparison via physical observables |
| `DATA/` | Small coarse-mesh tensors for local execution and testing |
| `RESULTS/` | High-resolution compression benchmarks, reconstruction error, band-gap renormalization drift |

**Note:** coarse data in `DATA/` is for execution tests only. Effective compression must be assessed on dense k-meshes as shown in `RESULTS/`.

---

## Design constraints

- **Diagram sampling speed is the bottleneck**, not DFT preprocessing
- Vertex lookup must not slow Monte-Carlo acceptance rates
- Compression must allow **on-the-fly reconstruction inside the sampler**
- Error is measured through **physical observables** (self-energy, gap renormalization), not Frobenius norms

---

## Scientific scope

- Finite-temperature electron–phonon self-energy
- Polaron formation study
- First-principles Hamiltonians embedded in stochastic field-theory solvers
- El-ph Tensor compression and inteprolation

---

