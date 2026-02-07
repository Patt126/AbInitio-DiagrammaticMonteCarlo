import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_energy_curve(path: Path):
    lambdas, e_over_t, avg_order = [], [], []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lambdas.append(float(row["lambda"]))
            e_over_t.append(float(row["energy_over_t"]))
            avg_order.append(float(row["avg_order"]))
    x = np.array(lambdas)
    y = np.array(e_over_t)
    n = np.array(avg_order)
    idx = np.argsort(x)
    return x[idx], y[idx], n[idx]


def load_green_curve(path: Path, target_lambda: float):
    tau, hist, green = [], [], []
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return np.array([]), np.array([]), np.array([]), None

    lambdas = np.array([float(r["lambda"]) for r in rows])
    unique_lambda = np.unique(lambdas)
    chosen = unique_lambda[np.argmin(np.abs(unique_lambda - target_lambda))]

    for r in rows:
        if abs(float(r["lambda"]) - chosen) < 1e-12:
            tau.append(float(r["tau"]))
            hist.append(float(r["histogram_data"]))
            green.append(float(r["green_estimator"]))
    tau = np.array(tau)
    hist = np.array(hist)
    green = np.array(green)
    idx = np.argsort(tau)
    return tau[idx], hist[idx], green[idx], float(chosen)


def main():
    parser = argparse.ArgumentParser(description="Plot breathing-coupling outputs.")
    parser.add_argument(
        "--energy-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "breathing_momentum.csv"),
    )
    parser.add_argument(
        "--green-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "breathing_green_momentum.csv"),
    )
    parser.add_argument("--lambda", dest="target_lambda", type=float, default=1.0)
    args = parser.parse_args()

    energy_csv = Path(args.energy_csv)
    green_csv = Path(args.green_csv)
    if not energy_csv.exists():
        raise FileNotFoundError(f"CSV not found: {energy_csv}. Run `make run-breathing` first.")
    if not green_csv.exists():
        raise FileNotFoundError(f"CSV not found: {green_csv}. Run `make run-breathing` first.")

    lam, e_over_t, avg_order = load_energy_curve(energy_csv)
    tau, hist, green, chosen_lambda = load_green_curve(green_csv, args.target_lambda)

    out_energy = energy_csv.with_suffix(".png")
    out_green = green_csv.with_name("breathing_green.png")

    plt.figure(figsize=(7.2, 5.0))
    plt.plot(lam, e_over_t, marker="o", linewidth=2, markersize=5, label="DiagMC")
    plt.xlabel(r"$\lambda = g^2/(\Omega t)$")
    plt.ylabel(r"$E_{GS}/t$")
    plt.title(r"Breathing coupling")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_energy, dpi=170)
    plt.close()

    plt.figure(figsize=(7.2, 5.0))
    plt.plot(tau, green, linewidth=2.2, label="Green estimator")
    plt.plot(tau, hist, "--", linewidth=1.8, label="Histogram")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$G(\tau)$")
    if chosen_lambda is None:
        plt.title("Breathing Green function")
    else:
        plt.title(f"Breathing Green function (lambda={chosen_lambda:.3g})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_green, dpi=170)
    plt.close()

    print(f"Saved: {out_energy}")
    print(f"Saved: {out_green}")


if __name__ == "__main__":
    main()
