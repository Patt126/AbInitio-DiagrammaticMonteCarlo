# Diagrammatic Monte Carlo Runs (Current Layout)

This project now has two standalone simulation entry points:

- `Benchmark/benchmark.cpp`
- `Breathing_Coupling/breathing.cpp`
- `sign_oscillation/sign_oscillation.cpp`

Both compile against the same solver core (`DMC.cpp`, `feynmanDiagram.cpp`).

## Build and run

From project root:

```bash
make benchmark
make breathing
make sign
```

Run simulations:

```bash
make run-benchmark
make run-breathing
make run-sign
```

Run both:

```bash
make run
```

## Output files

### Benchmark mode

- Main curve data: `Benchmark/results/benchmark.csv`
- Green-function data: `Benchmark/results/benchmark_green.csv`
- Plot script (manual): `Benchmark/plot_Benchmark.py`
  - default reads the two files above
  - writes:
    - `Benchmark/results/benchmark.png`
    - `Benchmark/results/benchmark_green.png`

### Breathing mode

- Main curve data: `Breathing_Coupling/results/breathing.csv`
- Green-function data: `Breathing_Coupling/results/breathing_green.csv`
- Plot script (manual): `Breathing_Coupling/plot_breathing.py`
  - default reads the two files above
  - writes:
    - `Breathing_Coupling/results/breathing.png`
    - `Breathing_Coupling/results/breathing_green.png`

### Sign oscillation demo

- Main curve data (two runs):
  - `sign_oscillation/results/sign_oscillation_abs.csv`
  - `sign_oscillation/results/sign_oscillation_signed.csv`
- Green-function data (two runs):
  - `sign_oscillation/results/sign_oscillation_abs_green.csv`
  - `sign_oscillation/results/sign_oscillation_signed_green.csv`
- Plot script (manual): `sign_oscillation/plot_sign_oscillation.py`
  - default reads the two files above
  - writes:
    - `sign_oscillation/results/sign_oscillation_green.png`

## Coupling formulas used

### 1) Benchmark mode

The simulation uses a momentum-dependent coupling:

\[
g(k,q)=g_0\left(1+0.25\cos q\right).
\]

Electron dispersion:

\[
\epsilon_k=-2t\cos k-\mu.
\]

### 2) Breathing-coupling mode

Paper form:

\[
g_q=-2ig\sin\left(\frac{q}{2}\right),\qquad
\lambda=\frac{g^2}{\Omega t}.
\]

In the current sign-free implementation we use the real magnitude in the Monte Carlo weight:

\[
\lvert g_q\rvert =2g\left\lvert\sin\left(\frac{q}{2}\right)\right\rvert.
\]

The sweep is performed in \(\lambda\), with \(g=\sqrt{\lambda\,\Omega t}\).

### 3) Sign oscillation demo

Signed coupling per vertex:

\[
g(q; m_{\mathrm{in}}, m_{\mathrm{out}}) = g_0 \sin\left[q a (m_{\mathrm{in}} - m_{\mathrm{out}})\right],
\]

and the sampling uses \(\lvert g \rvert\). The phase-reweighted run uses
\(g \to g\,e^{i q a (m_{\mathrm{in}}-m_{\mathrm{out}})}\) while keeping the same
\(|g|\) for sampling; the plot overlays the improved estimator for \(|g|\)
and the phase-reweighted real part.
