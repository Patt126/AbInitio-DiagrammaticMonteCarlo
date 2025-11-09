## Electron–Phonon Matrix Elements Post-Processing

Post-processing tools for analyzing and compressing DFPT electron–phonon matrix elements from **VASP 6.5.1** (`vaspelph.h5`).

### `matrix_element.ipynb`

Physics validation and inspection of the raw electron–phonon tensor:
- Consistency and symmetry analysis of coupling tensor structure
- Electronic bands and phonon dispersion plots
- Coupling amplitude visualized on band structure
- Temperature population checks (Bose–Einstein, Dulong–Petit regime)
- Band-gap renormalization (BGR) estimates from e–ph interaction

### `order_reduction.ipynb`

Data-driven compression of electron–phonon matrix elements based on:  
**L. Bernardi et al., Phys. Rev. Lett. 125, 256402 (2020)**

Includes:
- SVD compression of e–ph coupling matrices
- Symmetry reduction prior to decomposition
- Transformation to Wannier space via **Wannier90** unitary matrices
- Truncation and reconstruction back to **k-space**
- Compression validation using **band-gap renormalization** error analysis

> **Note:** Compression is meaningful for **dense k-meshes and high-dimensional tensors**.  
> `DATA/` contains small coarse-mesh examples that run locally.  
> `RESULTS/` includes reference plots from large tensors not suitable for local execution.
