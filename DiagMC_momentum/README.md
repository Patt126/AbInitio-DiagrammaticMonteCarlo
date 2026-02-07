# Diagrammatic Monte Carlo Runs

This project now has two standalone simulation entry points:

- `Breathing_Coupling/breathing.cpp`
- `sign_oscillation/sign_oscillation.cpp`

Both compile against the same solver core (`DMC.cpp`, `feynmanDiagram.cpp`).

## Build and run

From project root:

```bash
make breathing
make sign
```

Run simulations:

```bash
make run-breathing
make run-sign
```

Run both:

```bash
make run
```

## Output files

### Breathing mode

- Main curve data: `Breathing_Coupling/results/breathing.csv`
- Green-function data: `Breathing_Coupling/results/breathing_green.csv`
- Plot script (manual): `Breathing_Coupling/plot_breathing.py`
- Paper Comparison (manual): `Breathing_Coupling/plot_comparison.py`


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

### 1) Breathing-coupling mode

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

### 2) Sign oscillation demo

Coupling amplitude matches the breathing form:

\[
g(q)=2g\sin\left(\frac{q}{2}\right),\qquad g=\sqrt{\lambda\,\Omega t}
\]

Sampling uses \(\lvert g(q)\rvert\). The phase‑reweighted run multiplies by an
external phase:

\[
g \to g\,e^{i(\text{phase}\cdot q)}
\]

In the result file, plots are present showing how unstable the sampling is over different run with same paremeters.

