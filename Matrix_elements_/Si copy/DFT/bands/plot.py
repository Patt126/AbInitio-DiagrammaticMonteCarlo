import numpy as np
import py4vasp
import matplotlib.pyplot as plt

calc = py4vasp.Calculation.from_path(".")
d = calc.band.to_dict()

bands = np.array(d["bands"])          # (nk, nb) oppure (spin, nk, nb)
occ   = np.array(d["occupations"])
x     = np.array(d["kpoint_distances"])

# ---- gestisci spin automaticamente ----
if bands.ndim == 3:   # spin-polarized
    bands = bands[0]  # prendi spin-up (per Si/LiF non cambia)
    occ   = occ[0]

nk, nb = bands.shape

# ---- identifica bande di valenza ----
# una banda è "valence" se è occupata (occ ~ 1) in ALMENO un k-point
is_valence = np.any(occ > 0.5, axis=0)

valence_band_indices = np.where(is_valence)[0]

# ---- ordina le bande di valenza per energia massima (VBM) ----
vbm_energy_per_band = bands[:, valence_band_indices].max(axis=0)
order = np.argsort(vbm_energy_per_band)

# ---- prendi le TOP 3 ----
top3_vb = valence_band_indices[order][-3:]

print("Top 3 valence band indices:", top3_vb)

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(5.5,4.0))

for b in top3_vb:
    ax.plot(x, bands[:, b], lw=2)

ax.axhline(0.0, color="gray", lw=0.6)
ax.set_xlabel("k-path")
ax.set_ylabel("Energy (eV)")
ax.set_title("Top 3 Valence Bands")

plt.tight_layout()
plt.savefig("top3_valence_bands.png")
plt.close()
