"""Part 3.1: delta-hedging simulation under GBM.

Outputs (results/):
  hedge_sim_frequency.csv, hedge_sim_vol_mismatch.csv, hedge_sim_std_vs_rebalances.csv,
  hedge_sim_distributions.png, hedge_sim_gamma_approx.png, hedge_sim_std_vs_rebalances.png
"""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

from options_lab.hedging.simulation import (
    HedgeSimConfig, gamma_theta_approx, hedge_pnl, simulate_paths, summarize,
)
from options_lab.plotting import RESULTS, SERIES, TEXT_2, apply_style, save

BASE = HedgeSimConfig()
SPD = BASE.steps_per_day
FREQS = {"daily": SPD, "weekly": 5 * SPD, "no rebalancing": None}


def frequency_table(S):
    rows, pnls = [], {}
    for cost in (0.0, 2.0):
        for name, he in FREQS.items():
            out = hedge_pnl(S, BASE, he, cost_bps=cost)
            rows.append({"rebalancing": name, "cost_bps": cost, "n_hedges": out["n_rebalances"],
                         **summarize(out["pnl"]), "mean_cost": float(out["cost"].mean()),
                         "premium": out["premium"]})
            pnls[(name, cost)] = out["pnl"]
    # Reference row: the same short call with no hedge at all.
    out = hedge_pnl(S, BASE, 0)
    rows.append({"rebalancing": "unhedged (reference)", "cost_bps": 0.0, "n_hedges": 0,
                 **summarize(out["pnl"]), "mean_cost": 0.0, "premium": out["premium"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "hedge_sim_frequency.csv", index=False)
    return df, pnls


def vol_mismatch():
    rows, pts = [], {}
    for sr in (0.15, 0.20, 0.25):
        cfg = replace(BASE, sigma_real=sr)
        S = simulate_paths(cfg)
        pnl = hedge_pnl(S, cfg, SPD)["pnl"]
        approx = gamma_theta_approx(S, cfg, SPD)
        slope, intercept = np.polyfit(approx, pnl, 1)
        rows.append({"sigma_imp": cfg.sigma_imp, "sigma_real": sr, **summarize(pnl),
                     "approx_mean": float(approx.mean()),
                     "corr_actual_vs_approx": float(np.corrcoef(pnl, approx)[0, 1]),
                     "regression_slope": float(slope), "regression_intercept": float(intercept)})
        pts[sr] = (approx, pnl)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "hedge_sim_vol_mismatch.csv", index=False)
    return df, pts


def std_vs_rebalances(S):
    n = BASE.n_steps
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    rows = []
    for he in sorted(divisors, reverse=True):
        out = hedge_pnl(S, BASE, he)
        rows.append({"hedge_every_steps": he, "n_hedges": out["n_rebalances"], "std": float(out["pnl"].std(ddof=1))})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "hedge_sim_std_vs_rebalances.csv", index=False)
    return df


def main():
    apply_style()
    S = simulate_paths(BASE)
    freq, pnls = frequency_table(S)
    mism, pts = vol_mismatch()
    stdn = std_vs_rebalances(S)

    # Distributions: small multiples, one per rebalancing frequency.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    bins = np.linspace(-6, 4, 81)
    for ax, name in zip(axes, FREQS):
        ax.hist(pnls[(name, 0.0)], bins=bins, color=SERIES[0], alpha=0.85, label="no costs")
        ax.hist(pnls[(name, 2.0)], bins=bins, histtype="step", color=SERIES[1], linewidth=1.5,
                label="2 bps costs")
        row = freq[(freq.rebalancing == name) & (freq.cost_bps == 0)].iloc[0]
        ax.axvline(0, color=TEXT_2, linewidth=0.8)
        ax.set(title=f"{name}: std {row['std']:.2f}", xlabel="Hedged P&L at expiry ($ per call)")
    axes[0].set_ylabel("Paths")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Short 30-day ATM call, implied = realised = 20%, premium ${freq.premium.iloc[0]:.2f}",
                 x=0.01, ha="left", fontsize=11, color=TEXT_2)
    save(fig, "hedge_sim_distributions.png")

    # Actual vs gamma-theta approximation
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, (sr, (approx, pnl)) in zip(axes, pts.items()):
        k = slice(0, 3000)
        ax.scatter(approx[k], pnl[k], s=6, color=SERIES[0], alpha=0.35, linewidths=0)
        lim = np.array([-4, 3])
        ax.plot(lim, lim, color=TEXT_2, linewidth=1, label="45° line")
        row = mism[mism.sigma_real == sr].iloc[0]
        ax.set(title=f"realised vol {sr:.0%}\nmean P&L {row['mean']:+.2f}, corr {row.corr_actual_vs_approx:.3f}",
               xlabel="Σ ½ Γ S² (σ_imp² dt − r_t²)")
    axes[0].set_ylabel("Actual daily-hedged P&L ($)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Hedged P&L is the gamma-weighted gap between implied and realised variance (implied = 20%)",
                 x=0.01, y=1.06, ha="left", fontsize=11, color=TEXT_2)
    save(fig, "hedge_sim_gamma_approx.png")

    # Std vs number of hedges
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(stdn.n_hedges, stdn["std"], color=SERIES[0], marker="o", label="simulated std")
    ref = stdn["std"].iloc[-1] * np.sqrt(stdn.n_hedges.iloc[-1] / stdn.n_hedges)
    ax.loglog(stdn.n_hedges, ref, color=TEXT_2, linestyle="--", linewidth=1, label="∝ 1/√N")
    ax.set(title="Hedging error shrinks roughly like 1/√N",
           xlabel="Number of hedges over 30 days (log)", ylabel="Std of hedged P&L ($, log)")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.2, 0.3, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticks([1, 2, 5, 10, 21, 42, 84])
    ax.legend()
    save(fig, "hedge_sim_std_vs_rebalances.png")

    pd.set_option("display.width", 180)
    print(freq.round(4).to_string(index=False))
    print(mism.round(4).to_string(index=False))
    print(stdn.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
