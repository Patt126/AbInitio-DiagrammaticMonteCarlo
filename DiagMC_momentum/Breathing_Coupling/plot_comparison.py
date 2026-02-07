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

def load_paper_curve(path: Path):
    lambdas, e_over_t = [], []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lambdas.append(float(row["lambda"]))
            e_over_t.append(float(row["energy_over_t"]))
    x = np.array(lambdas)
    y = np.array(e_over_t)
    idx = np.argsort(x)
    return x[idx], y[idx]



def main():
    parser = argparse.ArgumentParser(description="Plot breathing-coupling outputs.")
    parser.add_argument(
        "--energy-csv_0",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "breathing_0.csv"),
    )
    parser.add_argument(
        "--energy-csv_q",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "breathing_q.csv"),
    )
    parser.add_argument(
        "--energy-csv_momentum",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "breathing_momentum.csv"),
    )
    parser.add_argument(
        "--energy-csv_paper",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "breathing_paper_MO0.csv"),
    )
 
    parser.add_argument("--lambda", dest="target_lambda", type=float, default=1.0)
    args = parser.parse_args()

    energy_csv_0 = Path(args.energy_csv_0)
    energy_csv_q = Path(args.energy_csv_q)
    energy_csv_momentum = Path(args.energy_csv_momentum)
    energy_csv_paper = Path(args.energy_csv_paper)

    if not energy_csv_paper.exists():
        raise FileNotFoundError(f"CSV not found: {energy_csv_paper}. Run `make run-breathing` first.")
    if not energy_csv_paper.exists():
        raise FileNotFoundError(f"CSV not found: {energy_csv_paper}. Run `make run-breathing` first.")

    lam_0, e_over_t_0, _ = load_energy_curve(energy_csv_0)
    lam_q, e_over_t_q, _ = load_energy_curve(energy_csv_q)
    lam_momentum, e_over_t_momentum, _ = load_energy_curve(energy_csv_momentum)
    lam_paper, e_over_t_paper= load_paper_curve(energy_csv_paper)

    out_energy =  energy_csv_paper.with_suffix(".png")


    plt.figure(figsize=(7.2, 5.0))
    plt.plot(lam_paper, e_over_t_paper, linewidth=1, label="MO0")
    plt.plot(lam_q, e_over_t_q, marker="o", linestyle="--", linewidth=1, markersize=3, label="g(q)-DiagMC")
    plt.plot(lam_momentum, e_over_t_momentum, marker="o", linestyle="--", linewidth=1, markersize=3, label="g(q)-DiagMC-momentum")
    plt.plot(lam_0[:11], e_over_t_0[:11], marker="o",linestyle="--", linewidth=1, markersize=3, label="g-DiagMC")
    

    plt.xlabel(r"$\lambda = g^2/(\Omega t)$")
    plt.ylabel(r"$E_{GS}/t$")
    plt.title(r"Breathing coupling")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_energy, dpi=170)
    plt.close()

    

    print(f"Saved: {out_energy}")



if __name__ == "__main__":
    main()
