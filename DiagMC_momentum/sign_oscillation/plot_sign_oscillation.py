import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_green(path: Path):
    tau, gvals, gerr = [], [], []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tau.append(float(row["tau"]))
            gvals.append(float(row["green_estimator"]))
            gerr.append(float(row.get("green_error", "0.0")))
    tau = np.array(tau)
    gvals = np.array(gvals)
    gerr = np.array(gerr)
    if tau.size == 0:
        return tau, gvals, gerr
    idx = np.argsort(tau)
    return tau[idx], gvals[idx], gerr[idx]


def main():
    parser = argparse.ArgumentParser(description="Plot sign-oscillation Green functions.")
    parser.add_argument(
        "--abs-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "sign_oscillation_abs_green.csv"),
    )
    parser.add_argument(
        "--signed-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "sign_oscillation_signed_green.csv"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "sign_oscillation_green.png"),
    )
    parser.add_argument("--tau-max", type=float, default=None)
    parser.add_argument("--renorm", action="store_true")
    parser.add_argument("--renorm-bins", type=int, default=1)
    args = parser.parse_args()

    abs_csv = Path(args.abs_csv)
    signed_csv = Path(args.signed_csv)
    if not abs_csv.exists():
        raise FileNotFoundError(f"Missing {abs_csv}. Run `make run-sign-abs` first.")
    if not signed_csv.exists():
        raise FileNotFoundError(f"Missing {signed_csv}. Run `make run-sign-signed` first.")

    tau_abs, g_abs, g_abs_err = load_green(abs_csv)
    tau_signed, g_signed, g_signed_err = load_green(signed_csv)

    if args.renorm:
        n = max(1, args.renorm_bins)
        if g_abs.size >= n:
            norm_abs = np.mean(g_abs[:n])
            if norm_abs != 0:
                g_abs = g_abs / norm_abs
                g_abs_err = g_abs_err / abs(norm_abs)
        if g_signed.size >= n:
            norm_signed = np.mean(g_signed[:n])
            if norm_signed != 0:
                g_signed = g_signed / norm_signed
                g_signed_err = g_signed_err / abs(norm_signed)

    if args.tau_max is not None:
        mask_abs = tau_abs <= args.tau_max
        tau_abs, g_abs, g_abs_err = tau_abs[mask_abs], g_abs[mask_abs], g_abs_err[mask_abs]
        mask_signed = tau_signed <= args.tau_max
        tau_signed, g_signed, g_signed_err = (
            tau_signed[mask_signed],
            g_signed[mask_signed],
            g_signed_err[mask_signed],
        )

    plt.figure(figsize=(7.4, 5.2))
    if tau_abs.size:
        plt.plot(tau_abs, g_abs, linewidth=2.2, label="|g| (abs)")
        if g_abs_err.size:
            plt.fill_between(
                tau_abs,
                g_abs - g_abs_err,
                g_abs + g_abs_err,
                color="tab:blue",
                alpha=0.3,
                linewidth=0.0,
            )
    if tau_signed.size:
        plt.plot(tau_signed, g_signed, linewidth=2.2, label="phase-reweighted (Re)", alpha=0.9)
        if g_signed_err.size:
            plt.fill_between(
                tau_signed,
                g_signed - g_signed_err,
                g_signed + g_signed_err,
                color="tab:orange",
                alpha=0.3,
                linewidth=0.0,
            )
    plt.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    plt.grid(alpha=0.25)
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$G(\tau)$ (improved estimator)")
    plt.title("Sign oscillation: abs vs phase-reweighted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=170)
    plt.close()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
